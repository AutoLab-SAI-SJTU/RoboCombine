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
logger = logging.getLogger(__name__)

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    import numpy as np
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

@dataclass
class ActionExpertConfig:
    dim: int = 1024
    ffn_dim: int = 4096
    num_layers: int = 30
    state_dim: int = 14
    action_dim: int = 14
    chunk_size: int = 16
    video_feature_dim: int = 3072
    causal: bool = False
    num_registers: int = 4
    eps: float = 1e-06
    training_mode: str = 'finetune'

    def __post_init__(self):
        assert self.chunk_size >= 2, 'chunk_size must be at least 2 (1 state + 1 action)'

class StateActionEncoder(nn.Module):

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config
        self.state_encoder = self.build_mlp('mlp3x_silu', in_features=config.state_dim, out_features=config.dim)
        self.action_encoder = self.build_mlp('mlp3x_silu', in_features=config.action_dim, out_features=config.dim)
        max_seq_len = config.chunk_size + 1 + config.num_registers
        pos_embed = get_1d_sincos_pos_embed_from_grid(config.dim, np.arange(max_seq_len))
        self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))

    def build_mlp(self, projector_type, in_features, out_features):
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

    def forward(self, state_tokens: torch.Tensor, action_tokens: torch.Tensor, registers: torch.Tensor=None) -> torch.Tensor:
        B = state_tokens.shape[0]
        chunk_size = state_tokens.shape[1] + action_tokens.shape[1]
        state_encoded = self.state_encoder(state_tokens)
        action_encoded = self.action_encoder(action_tokens)
        encoded = torch.cat([state_encoded, action_encoded], dim=1)
        if registers is not None:
            encoded = torch.cat([encoded, registers], dim=1)
        seq_len = encoded.shape[1]
        encoded = encoded + self.pos_embedding[:, :seq_len, :]
        return encoded

class ActionEncoder(nn.Module):

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config
        self.action_encoder = self.build_mlp('mlp3x_silu', in_features=config.action_dim, out_features=config.dim)
        max_seq_len = config.chunk_size + config.num_registers
        pos_embed = get_1d_sincos_pos_embed_from_grid(config.dim, np.arange(max_seq_len))
        self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))

    def build_mlp(self, projector_type, in_features, out_features):
        if projector_type == 'linear':
            return nn.Linear(in_features, out_features)
        mlp_silu_match = re.match('^mlp(\\d+)x_silu$', projector_type)
        if mlp_silu_match:
            mlp_depth = int(mlp_silu_match.group(1))
            modules = [nn.Linear(in_features, out_features)]
            for _ in range(1, mlp_depth):
                modules.append(nn.SiLU())
                modules.append(nn.Linear(out_features, out_features))
            return nn.Sequential(*modules)
        raise ValueError(f'Unknown projector type: {projector_type}')

    def forward(self, state_tokens: torch.Tensor, action_tokens: torch.Tensor, registers: torch.Tensor=None) -> torch.Tensor:
        action_encoded = self.action_encoder(action_tokens)
        encoded = action_encoded
        if registers is not None:
            encoded = torch.cat([encoded, registers], dim=1)
        seq_len = encoded.shape[1]
        encoded = encoded + self.pos_embedding[:, :seq_len, :]
        return encoded

class ActionExpertBlock(nn.Module):

    def __init__(self, config: ActionExpertConfig, wan_config: dict):
        super().__init__()
        self.config = config
        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)
        self.wan_num_heads = wan_config['num_heads']
        self.wan_head_dim = wan_config['head_dim']
        self.wan_dim = wan_config['dim']
        assert self.wan_num_heads * self.wan_head_dim == self.wan_dim
        self.wan_action_qkv = nn.Parameter(torch.randn(3, self.wan_num_heads, config.dim, self.wan_head_dim) / (config.dim * self.wan_head_dim) ** 0.5)
        self.wan_action_o = nn.Linear(self.wan_dim, config.dim, bias=False)
        self.wan_action_norm_q = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.wan_action_norm_k = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.ffn = nn.Sequential(nn.Linear(config.dim, config.ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(config.ffn_dim, config.dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, config.dim) / config.dim ** 0.5)

class ActionDecoder(nn.Module):

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config
        self.norm = WanLayerNorm(config.dim, eps=config.eps)
        self.action_head = self.build_mlp('mlp1x_silu', config.dim, config.action_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, config.dim) / config.dim ** 0.5)

    def build_mlp(self, projector_type, in_features, out_features):
        if projector_type == 'linear':
            return nn.Linear(in_features, out_features)
        mlp_silu_match = re.match('^mlp(\\d+)x_silu$', projector_type)
        if mlp_silu_match:
            mlp_depth = int(mlp_silu_match.group(1))
            modules = [nn.Linear(in_features, out_features)]
            for _ in range(1, mlp_depth):
                modules.append(nn.SiLU())
                modules.append(nn.Linear(out_features, out_features))
            return nn.Sequential(*modules)
        raise ValueError(f'Unknown projector type: {projector_type}')

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e0, e1 = (self.modulation.unsqueeze(0) + time_emb.unsqueeze(2)).chunk(2, dim=2)
        z = self.norm(x) * (1 + e1.squeeze(2)) + e0.squeeze(2)
        return self.action_head(z)

class ActionExpert(nn.Module):

    def __init__(self, config: ActionExpertConfig, wan_config: dict=None):
        super().__init__()
        self.config = config
        self.freq_dim = 256
        if config.training_mode == 'pretrain':
            self.input_encoder = ActionEncoder(config)
        else:
            self.input_encoder = StateActionEncoder(config)
        self.time_embedding = nn.Sequential(nn.Linear(self.freq_dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, config.dim * 6))
        if wan_config is not None:
            self.blocks = nn.ModuleList([ActionExpertBlock(config, wan_config) for _ in range(config.num_layers)])
        else:
            self.blocks = nn.ModuleList([ActionExpertBlock(config, {'dim': 3072, 'num_heads': 24, 'head_dim': 128}) for _ in range(config.num_layers)])
        if config.num_registers > 0:
            self.registers = nn.Parameter(torch.empty(1, config.num_registers, config.dim).normal_(std=0.02))
        else:
            self.registers = None
        self.decoder = ActionDecoder(config)
        self.initialize_weights()
        logger.info(f'Action Expert initialized with {self.count_parameters():,} parameters')

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.decoder.action_head[-1].weight)
        nn.init.zeros_(self.decoder.action_head[-1].bias)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def count_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters()))
