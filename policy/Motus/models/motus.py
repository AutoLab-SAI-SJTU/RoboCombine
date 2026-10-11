import sys
import json
import torch
import logging
import torch.nn as nn
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
BAK_ROOT = str((Path(__file__).parent.parent / 'bak').resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)
from utils.common import get_t_distribution
from wan.modules.model import sinusoidal_embedding_1d
from transformers import AutoConfig
from transformers import AutoModelForImageTextToText
from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .und_expert import UndExpert, UndExpertConfig
from wan.utils.fm import FlowMatchScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
logger = logging.getLogger(__name__)

@dataclass
class MotusConfig:
    wan_checkpoint_path: str = ''
    vae_path: str = ''
    wan_config_path: str = ''
    video_precision: str = 'bfloat16'
    vlm_checkpoint_path: str = ''
    und_expert_hidden_size: int = 512
    und_expert_ffn_dim_multiplier: int = 4
    und_expert_norm_eps: float = 1e-05
    und_layers_to_extract: List[int] = None
    vlm_adapter_input_dim: int = 2048
    vlm_adapter_projector_type: str = 'mlp3x_silu'
    num_layers: int = 30
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024
    action_expert_ffn_dim_multiplier: int = 4
    action_expert_norm_eps: float = 1e-06
    global_downsample_rate: int = 3
    video_action_freq_ratio: int = 4
    num_video_frames: int = 4
    video_height: int = 512
    video_width: int = 512
    batch_size: int = 8
    training_mode: str = 'finetune'
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    load_pretrained_backbones: Optional[bool] = None
    device: str = 'cuda'

    def __post_init__(self):
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        if self.und_layers_to_extract is None:
            self.und_layers_to_extract = list(range(self.num_layers))

class VideoModule(nn.Module):

    def __init__(self, video_model, dtype, device, grid_sizes):
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)
        video_features = video_patched.flatten(2).transpose(1, 2)
        return video_features

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        if isinstance(language_embeddings, list):
            text_len = self.video_model.wan_model.text_len
            padded_embeddings = []
            for emb in language_embeddings:
                if emb.shape[0] <= text_len:
                    padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                else:
                    padded = emb[:text_len]
                padded_embeddings.append(padded)
            t5_context_raw = torch.stack(padded_embeddings, dim=0)
        else:
            t5_context_raw = language_embeddings
        t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)
        return t5_context

    def get_time_embedding(self, t_video: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()
            t_emb = self.video_model.wan_model.time_embedding(sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float())
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, 3072))
            assert t_emb.dtype == torch.float32 and t_emb_proj.dtype == torch.float32
        return (t_emb, t_emb_proj)

    def process_cross_attention(self, video_tokens: torch.Tensor, video_adaln_params: torch.Tensor, layer_idx: int, processed_t5_context: torch.Tensor) -> torch.Tensor:
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        context_lens = None
        cross_out = wan_layer.cross_attn(wan_layer.norm3(video_tokens), processed_t5_context, context_lens)
        return video_tokens + cross_out

    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (wan_layer.modulation.unsqueeze(0) + video_adaln_params).chunk(6, dim=2)
        return modulation

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        v_mod = video_adaln_modulation
        ffn_input = wan_layer.norm2(video_tokens).float() * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)
        ffn_out = wan_layer.ffn(ffn_input)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(self, video_tokens: torch.Tensor, video_time_emb: torch.Tensor) -> torch.Tensor:
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)
        x = self.video_model.wan_model.unpatchify(x, self.grid_sizes)
        return torch.stack([u.float() for u in x], dim=0)

    def process_joint_attention(self, video_tokens: torch.Tensor, action_tokens: torch.Tensor, video_adaln_modulation: tuple, action_adaln_modulation: tuple, layer_idx: int, action_block: nn.Module, und_tokens: torch.Tensor, und_block: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)
        norm_action = action_block.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)
        B, L_v, C = norm_video.shape
        L_a = norm_action.shape[1]
        n = self.video_model.wan_model.num_heads
        d = C // n
        a_qkv = torch.einsum('BTD,KNDE->KBTNE', norm_action, action_block.wan_action_qkv)
        a_q_h, a_k_h, a_v_h = (a_qkv[0], a_qkv[1], a_qkv[2])
        a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)
        norm_und = und_block.norm1(und_tokens)
        L_u = norm_und.shape[1]
        u_qkv = torch.einsum('BTD,KNDE->KBTNE', norm_und, und_block.wan_und_qkv)
        u_q_h, u_k_h, u_v_h = (u_qkv[0], u_qkv[1], u_qkv[2])
        u_q = und_block.wan_und_norm_q(u_q_h.flatten(-2)).view(B, L_u, n, d)
        u_k = und_block.wan_und_norm_k(u_k_h.flatten(-2)).view(B, L_u, n, d)
        u_v = u_v_h.view(B, L_u, n, d)
        seq_lens = torch.full((B,), L_v + L_a + L_u, dtype=torch.long, device=self.device)
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)
        y, action_out_h, und_out_h = wan_layer.self_attn(norm_video, seq_lens, self.grid_sizes, freqs, action_q=a_q, action_k=a_k, action_v=a_v, und_q=u_q, und_k=u_k, und_v=u_v)
        und_out = und_block.wan_und_o(und_out_h.flatten(2))
        action_out = action_block.wan_action_o(action_out_h.flatten(2))
        video_tokens = video_tokens + y * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
        und_tokens = und_tokens + und_out
        return (video_tokens, action_tokens, und_tokens)

class UndModule(nn.Module):

    def __init__(self, vlm_model, und_expert, config, dtype, device):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        self.vlm_model = vlm_model
        self.und_expert = und_expert

    def extract_und_features(self, vlm_inputs) -> torch.Tensor:
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, B)
        vlm_kwargs = {'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask, 'position_ids': position_ids, 'past_key_values': None, 'use_cache': False, 'output_attentions': False, 'output_hidden_states': True, 'return_dict': True}
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds
        with torch.no_grad():
            vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)
        last_layer_features = vlm_output.hidden_states[-1]
        adapted_features = self.und_expert.vlm_adapter(last_layer_features)
        return adapted_features

    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        if isinstance(vlm_inputs, list):
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
            max_seq_len = max((ids.shape[1] for ids in input_ids_list))
            padded_input_ids = []
            padded_attention_masks = []
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)
        else:
            input_ids_batch = vlm_inputs['input_ids'].to(self.device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(self.device)
            pixel_values_batch = vlm_inputs['pixel_values'].to(self.device)
            image_grid_thw_batch = vlm_inputs['image_grid_thw'].to(self.device)
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)
        image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(pixel_values_batch, image_grid_thw_batch)
        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)
        image_mask, _ = self.vlm_model.model.get_placeholder_mask(input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        position_ids, _rope_deltas = self.vlm_model.model.get_rope_index(input_ids=input_ids_batch, image_grid_thw=image_grid_thw_batch, video_grid_thw=None, attention_mask=attention_mask_batch)
        return (inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids)

    def process_ffn(self, und_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        block = self.und_expert.blocks[layer_idx]
        ffn_input = block.norm2(und_tokens)
        ffn_output = block.ffn(ffn_input)
        und_tokens = und_tokens + ffn_output
        return und_tokens

class ActionModule(nn.Module):

    def __init__(self, action_expert: ActionExpert, config, video_model, vlm_model, dtype, device):
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model
        self.vlm_model = vlm_model
        self.dtype = dtype
        self.device = device

    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            a_e = self.action_expert.time_embedding(sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float())
            a_e0 = self.action_expert.time_projection(a_e).unflatten(2, (6, self.config.action_expert_dim))
            assert a_e.dtype == torch.float32 and a_e0.dtype == torch.float32
        return (a_e, a_e0)

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (action_layer.modulation.unsqueeze(0) + action_adaln_params).chunk(6, dim=2)
        return modulation

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        action_block = self.action_expert.blocks[layer_idx]
        a_mod = action_adaln_modulation
        ffn_input = action_block.norm2(action_tokens).float() * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)
        ffn_out = action_block.ffn(ffn_input)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_tokens = action_tokens + ffn_out * a_mod[5].squeeze(2)
        return action_tokens

class Motus(nn.Module):

    def __init__(self, config: MotusConfig):
        super().__init__()
        self.config = config
        self.dtype = torch.bfloat16
        self.device = getattr(config, 'device', 'cuda')
        load_backbones = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)
        logger.info('Initializing WAN video model...')
        if load_backbones:
            self.video_model = WanVideoModel.from_pretrained(checkpoint_path=config.wan_checkpoint_path, vae_path=config.vae_path, config_path=config.wan_config_path, precision=config.video_precision)
        else:
            self.video_model = WanVideoModel.from_config(config_path=config.wan_config_path, vae_path=config.vae_path, device=self.device, precision=config.video_precision)
        logger.info('Initializing VLM (frozen)...')
        if load_backbones:
            self.vlm_model = AutoModelForImageTextToText.from_pretrained(config.vlm_checkpoint_path, dtype=self.dtype, device_map='cuda', trust_remote_code=True)
        else:
            vlm_cfg = AutoConfig.from_pretrained(config.vlm_checkpoint_path, trust_remote_code=True)
            self.vlm_model = AutoModelForImageTextToText.from_config(vlm_cfg, torch_dtype=self.dtype, trust_remote_code=True)
            self.vlm_model.to(device=self.device, dtype=self.dtype)
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        logger.info('VLM parameters frozen')
        logger.info(f'VLM kept complete with {len(self.vlm_model.model.language_model.layers)} layers')
        wan_dim = getattr(self.video_model.wan_model.config, 'dim', 3072)
        wan_num_heads = getattr(self.video_model.wan_model.config, 'num_heads', 24)
        wan_head_dim = wan_dim // wan_num_heads
        vlm_dim = self.vlm_model.config.text_config.hidden_size
        vlm_num_heads = self.vlm_model.config.text_config.num_attention_heads
        vlm_num_kv_heads = getattr(self.vlm_model.config.text_config if hasattr(self.vlm_model.config, 'text_config') else self.vlm_model.config, 'num_key_value_heads', vlm_num_heads)
        vlm_num_hidden_layers = self.vlm_model.config.text_config.num_hidden_layers
        vlm_head_dim = vlm_dim // vlm_num_heads
        logger.info(f'Model configurations:')
        logger.info(f'  WAN: {wan_num_heads} heads × {wan_head_dim} head_dim = {wan_dim}D')
        logger.info(f'  VLM: {vlm_num_heads} Q heads, {vlm_num_kv_heads} KV heads × {vlm_head_dim} head_dim = {vlm_dim}D')
        wan_config = {'dim': wan_dim, 'num_heads': wan_num_heads, 'head_dim': wan_head_dim}
        vlm_config = {'hidden_size': vlm_dim, 'num_attention_heads': vlm_num_heads, 'num_key_value_heads': vlm_num_kv_heads, 'head_dim': vlm_head_dim, 'num_hidden_layers': vlm_num_hidden_layers}
        logger.info('Initializing Action Expert...')
        if config.training_mode == 'pretrain':
            action_chunk_size_for_expert = config.action_chunk_size
        else:
            action_chunk_size_for_expert = config.action_chunk_size + 1
        num_registers = 0 if config.training_mode == 'pretrain' else 4
        action_config = ActionExpertConfig(dim=config.action_expert_dim, ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier, num_layers=config.num_layers, state_dim=config.action_state_dim, action_dim=config.action_dim, chunk_size=action_chunk_size_for_expert, num_registers=num_registers, video_feature_dim=wan_dim, causal=False, eps=config.action_expert_norm_eps, training_mode=config.training_mode)
        self.action_expert = ActionExpert(action_config, wan_config)
        logger.info('Initializing Understanding Expert...')
        und_config = UndExpertConfig(dim=config.und_expert_hidden_size, ffn_dim=config.und_expert_hidden_size * config.und_expert_ffn_dim_multiplier, num_layers=config.num_layers, vlm_input_dim=config.vlm_adapter_input_dim, vlm_projector_type=config.vlm_adapter_projector_type, eps=config.und_expert_norm_eps)
        self.und_expert = UndExpert(und_config, wan_config, vlm_config)
        self.action_expert.to(device=self.device, dtype=self.dtype)
        self.und_expert.to(device=self.device, dtype=self.dtype)
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)
        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        batch_size = config.batch_size
        self.grid_sizes = torch.tensor([lat_T, lat_H, lat_W], dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        logger.info(f'Pre-computed grid_sizes: T={lat_T}, H={lat_H}, W={lat_W}')
        self.video_module = VideoModule(self.video_model, self.dtype, self.device, self.grid_sizes)
        self.und_module = UndModule(self.vlm_model, self.und_expert, self.config, self.dtype, self.device)
        self.action_module = ActionModule(self.action_expert, self.config, self.video_model, self.vlm_model, self.dtype, self.device)
        time_dist_config = getattr(config, 'time_distribution', {})
        model_config = {'timestep_sample_method': time_dist_config.get('timestep_sample_method', 'logit_normal'), 'sigmoid_scale': time_dist_config.get('sigmoid_scale', 1.0), 'min_t': time_dist_config.get('min_t', 0.0), 'max_t': time_dist_config.get('max_t', 1.0)}
        try:
            self.fm_train_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
            self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            logger.info('Initialized FlowMatchScheduler for training (video)')
        except Exception as e:
            logger.warning(f'Failed to init FlowMatchScheduler: {e}')
        try:
            self.fm_train_scheduler_action = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
            self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
            logger.info('Initialized FlowMatchScheduler for training (action)')
        except Exception as e:
            logger.warning(f'Failed to init FlowMatchScheduler for action: {e}')
        self.log_parameter_counts()

    def log_parameter_counts(self):
        total_params = sum((p.numel() for p in self.parameters()))
        trainable_params = sum((p.numel() for p in self.parameters() if p.requires_grad))
        video_params = sum((p.numel() for p in self.video_model.parameters()))
        action_params = sum((p.numel() for p in self.action_expert.parameters()))
        vlm_params = sum((p.numel() for p in self.vlm_model.parameters()))
        und_params = sum((p.numel() for p in self.und_expert.parameters()))
        logger.info(f'Motus parameter breakdown:')
        logger.info(f'  Total parameters: {total_params / 1000000000.0:.2f}B')
        logger.info(f'  Trainable parameters: {trainable_params / 1000000000.0:.2f}B')
        logger.info(f'  Video Model (WAN): {video_params / 1000000000.0:.2f}B')
        logger.info(f'  Action Expert: {action_params / 1000000.0:.1f}M')
        logger.info(f'  VLM (frozen): {vlm_params / 1000000000.0:.2f}B')
        logger.info(f'  Und Expert: {und_params / 1000000.0:.1f}M')

    def load_checkpoint(self, path: str, strict: bool=True) -> Dict:
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / 'mp_rank_00_model_states.pt'
            if not checkpoint_file.exists():
                raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_file}')
            path = str(checkpoint_file)
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['module']
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        logger.info(f'Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}')
        additional_state = {k: v for k, v in checkpoint.items() if k not in ['module', 'config']}
        return additional_state

    def load_pretrain_weights(self, path: str) -> None:
        if self.config.training_mode != 'finetune':
            raise ValueError('load_pretrain_weights should be called only in finetune mode')
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / 'pytorch_model' / 'mp_rank_00_model_states.pt'
            if not checkpoint_file.exists():
                raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_file}')
            path = str(checkpoint_file)
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('module', checkpoint)
        filtered = {}
        for k, v in state_dict.items():
            if 'action_expert.input_encoder' in k or 'action_expert.decoder' in k:
                continue
            filtered[k] = v
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        logger.info(f'Loaded pretrain weights (filtered). Missing: {len(missing)}, Unexpected: {len(unexpected)}')

    def training_step(self, first_frame: torch.Tensor, video_frames: torch.Tensor, state: torch.Tensor=None, actions: torch.Tensor=None, language_embeddings: Optional[List[torch.Tensor]]=None, vlm_inputs: Optional[List]=None, return_dict: bool=True) -> Dict[str, torch.Tensor]:
        B = video_frames.shape[0]
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)
        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))
        timestep_id = torch.randint(0, self.fm_train_scheduler.num_train_timesteps, (B,))
        video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))
        timestep_id_action = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (B,))
        action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(dtype=self.dtype, device=self.device)
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(dtype=self.dtype, device=self.device).view(B, 1, 1)
        action_noise = torch.randn_like(actions, dtype=self.dtype)
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action
        action_target = action_noise - actions
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)
            action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_embed, video_tokens.shape[1])
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_embed, action_tokens.shape[1])
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)
        with torch.autocast(device_type='cuda', dtype=self.video_model.precision):
            for layer_idx in range(self.config.num_layers):
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, self.action_expert.blocks[layer_idx], und_tokens, self.und_expert.blocks[layer_idx])
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)
            video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
            up_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
            if self.config.training_mode == 'pretrain':
                action_pred = action_pred_full[:, :up_len, :]
            else:
                action_pred = action_pred_full[:, 1:up_len, :]
            video_pred_masked = video_pred.clone()
            video_pred_masked[:, :, 0:1] = 0
            video_loss = torch.nn.functional.mse_loss(video_pred_masked, video_target, reduction='mean')
            action_loss = torch.nn.functional.mse_loss(action_pred, action_target, reduction='mean')
        total_loss = self.config.video_loss_weight * video_loss + self.config.action_loss_weight * action_loss
        if return_dict:
            return {'total_loss': total_loss, 'video_loss': video_loss, 'action_loss': action_loss, 'video_timestep_mean': sigma.float().mean().item(), 'action_timestep_mean': sigma_action.float().mean().item()}

    def _single_denoise_step(self, video_latent, action_latent, state, t, processed_t5_context, vlm_inputs, B):
        video_t_scaled = (t * 1000).expand(B).to(self.dtype)
        action_t_scaled = (t * 1000).expand(B).to(self.dtype)
        video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
        state_tokens = state.unsqueeze(1).to(self.dtype)
        registers = self.action_expert.registers.expand(B, -1, -1)
        action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        with torch.autocast(device_type='cuda', dtype=self.video_model.precision):
            video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_scaled, video_tokens.shape[1])
            action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_scaled, action_tokens.shape[1])
            for layer_idx in range(self.config.num_layers):
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, self.action_expert.blocks[layer_idx], und_tokens, self.und_expert.blocks[layer_idx])
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)
            video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
            action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
            action_velocity = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]
        return (video_velocity, action_velocity)

    def inference_step(self, first_frame: torch.Tensor, state: torch.Tensor=None, num_inference_steps: int=50, language_embeddings: Optional[List[torch.Tensor]]=None, vlm_inputs: Optional[List]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B = first_frame.shape[0]
        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        for i in range(num_inference_steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - t
            video_velocity, action_velocity = self._single_denoise_step(video_latent, action_latent, state, t, processed_t5_context, vlm_inputs, B)
            video_latent = video_latent + video_velocity * dt
            action_latent = action_latent + action_velocity * dt
            video_latent[:, :, 0:1] = condition_frame_latent
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]
            predicted_frames = (predicted_frames + 1.0) / 2.0
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
        predicted_actions = action_latent.float()
        return (predicted_frames, predicted_actions)
