import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
import math
import numpy as np
from dataclasses import dataclass
import logging
import sys
import re
from pathlib import Path
project_root = Path(__file__).parent.parent
bak_root = project_root / 'bak'
if str(bak_root.resolve()) not in sys.path:
    sys.path.insert(0, str(bak_root.resolve()))
from wan.modules.attention import flash_attention
from wan.modules.model import WanRMSNorm, WanLayerNorm, sinusoidal_embedding_1d, rope_apply
from utils.common import get_nd_sincos_pos_embed_from_grid
logger = logging.getLogger(__name__)

@dataclass
class UndExpertConfig:
    dim: int = 512
    ffn_dim: int = 2048
    num_layers: int = 30
    vlm_input_dim: int = 2048
    vlm_projector_type: str = 'mlp3x_silu'
    eps: float = 1e-05

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return torch.from_numpy(emb).float()

class UndExpertBlock(nn.Module):

    def __init__(self, config: UndExpertConfig, wan_config: dict):
        super().__init__()
        self.config = config
        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)
        self.wan_num_heads = wan_config['num_heads']
        self.wan_head_dim = wan_config['head_dim']
        self.wan_dim = wan_config['dim']
        assert self.wan_num_heads * self.wan_head_dim == self.wan_dim
        self.wan_und_qkv = nn.Parameter(torch.randn(3, self.wan_num_heads, config.dim, self.wan_head_dim) / (config.dim * self.wan_head_dim) ** 0.5)
        self.wan_und_o = nn.Linear(self.wan_dim, config.dim, bias=False)
        self.wan_und_norm_q = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.wan_und_norm_k = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.ffn = nn.Sequential(nn.Linear(config.dim, config.ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(config.ffn_dim, config.dim))

class UndExpert(nn.Module):

    def __init__(self, config: UndExpertConfig, wan_config: dict=None, vlm_config: dict=None):
        super().__init__()
        self.config = config
        self.freq_dim = 256
        self.vlm_adapter = self.build_condition_adapter(config.vlm_projector_type, config.vlm_input_dim, config.dim)
        if wan_config is not None:
            self.blocks = nn.ModuleList([UndExpertBlock(config, wan_config) for _ in range(config.num_layers)])
        else:
            self.blocks = nn.ModuleList([UndExpertBlock(config, {'dim': 3072, 'num_heads': 24, 'head_dim': 128}) for _ in range(config.num_layers)])

    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_silu_match = re.match('^mlp(\\d+)x_silu$', projector_type)
            if mlp_silu_match:
                mlp_depth = int(mlp_silu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.SiLU())
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)
        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')
        return projector
