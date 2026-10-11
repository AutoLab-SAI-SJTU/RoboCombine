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

    def _normalize_progress_language_embeddings(self, language_embeddings: Optional[List[torch.Tensor]]) -> List[torch.Tensor]:
        if language_embeddings is None:
            raise ValueError('language_embeddings must not be None for progress feature extraction')
        normalized = []
        for emb in language_embeddings:
            normalized.append(emb.to(self.device).to(self.dtype))
        return normalized

    def _build_progress_proxy_video_tokens(self, first_frame: torch.Tensor) -> torch.Tensor:
        first_frame = first_frame.to(self.device).to(self.dtype)
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm)
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        proxy_video_latent = condition_frame_latent.expand(-1, -1, num_total_latent_frames, -1, -1).contiguous()
        return self.video_module.prepare_input(proxy_video_latent.to(self.dtype))

    def _pool_action_tokens_for_progress(self, action_tokens: torch.Tensor) -> torch.Tensor:
        register_count = int(getattr(self.action_expert.config, 'num_registers', 0))
        token_end = action_tokens.shape[1] - register_count if register_count > 0 else action_tokens.shape[1]
        usable_tokens = action_tokens[:, :token_end, :]
        if usable_tokens.shape[1] == 0:
            raise ValueError('Action tokens for progress pooling are empty')
        if self.config.training_mode == 'pretrain':
            return usable_tokens.mean(dim=1)
        state_token = usable_tokens[:, 0, :]
        if usable_tokens.shape[1] == 1:
            action_mean = state_token
        else:
            action_mean = usable_tokens[:, 1:, :].mean(dim=1)
        return torch.cat([state_token, action_mean], dim=-1)

    @torch.no_grad()
    def extract_action_progress_features(self, first_frame: torch.Tensor, state: torch.Tensor, action_sequence: torch.Tensor, language_embeddings: Optional[List[torch.Tensor]]=None, vlm_inputs: Optional[Dict[str, torch.Tensor]]=None, progress_timestep: float=0.02) -> torch.Tensor:
        if state is None:
            raise ValueError('state must not be None for action progress feature extraction')
        if action_sequence is None:
            raise ValueError('action_sequence must not be None for action progress feature extraction')
        if vlm_inputs is None:
            raise ValueError('vlm_inputs must not be None for action progress feature extraction')
        B = first_frame.shape[0]
        language_embeddings = self._normalize_progress_language_embeddings(language_embeddings)
        state = state.to(self.device).to(self.dtype)
        action_sequence = action_sequence.to(self.device).to(self.dtype)
        video_tokens = self._build_progress_proxy_video_tokens(first_frame)
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, action_sequence, registers)
        else:
            state_tokens = state.unsqueeze(1)
            action_tokens = self.action_expert.input_encoder(state_tokens, action_sequence, registers)
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)
        timestep = torch.full((B,), float(progress_timestep), device=self.device, dtype=self.dtype)
        use_cuda_autocast = str(self.device).startswith('cuda')
        with torch.autocast(device_type='cuda', dtype=self.video_model.precision) if use_cuda_autocast else nullcontext():
            video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(timestep, video_tokens.shape[1])
            action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(timestep, action_tokens.shape[1])
            for layer_idx in range(self.config.num_layers):
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, self.action_expert.blocks[layer_idx], und_tokens, self.und_expert.blocks[layer_idx])
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)
        return self._pool_action_tokens_for_progress(action_tokens).float()

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

    def inference_step(self, first_frame: torch.Tensor, state: torch.Tensor=None, num_inference_steps: int=50, language_embeddings: Optional[List[torch.Tensor]]=None, vlm_inputs: Optional[List]=None, prior_action: Optional[torch.Tensor]=None, forcing_steps: int=0) -> Tuple[torch.Tensor, torch.Tensor]:
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
            if prior_action is not None and forcing_steps > 0:
                t_next = timesteps[i + 1]
                noise = torch.randn_like(prior_action[:, :forcing_steps, :])
                forced = (1 - t_next) * prior_action[:, :forcing_steps, :].to(self.dtype) + t_next * noise
                action_latent[:, :forcing_steps, :] = forced
            video_latent[:, :, 0:1] = condition_frame_latent
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]
            predicted_frames = (predicted_frames + 1.0) / 2.0
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
        predicted_actions = action_latent.float()
        return (predicted_frames, predicted_actions)

    def inference_step_cfg(self, first_frame: torch.Tensor, state: torch.Tensor=None, num_inference_steps: int=50, language_embeddings: Optional[List[torch.Tensor]]=None, vlm_inputs: Optional[List]=None, uncond_language_embeddings: Optional[List[torch.Tensor]]=None, uncond_vlm_inputs: Optional[List]=None, uncond_first_frame: Optional[torch.Tensor]=None, source_language_embeddings: Optional[List[torch.Tensor]]=None, source_vlm_inputs: Optional[List]=None, source_first_frame: Optional[torch.Tensor]=None, cfg_reference_mode: str='runtime_uncond', lambda_plus: float=1.0, lambda_minus: float=0.0, event_gate_beta: float=10.0, event_gate_threshold: float=0.25, event_gate_stop_threshold: float=0.5, conflict_cos_threshold: float=0.0, conflict_mode: str='negative_parallel_only', gate_mode: str='', gate_on_threshold: float=0.25, gate_off_threshold: float=0.15, gate_ema_decay: float=0.7, initial_gate_state: bool=False, initial_gate_ema_value: float=0.0, source_memory_decay: float=0.7, source_memory_mode: str='off', soft_mask_temperature: float=8.0, soft_mask_dominance_eta: float=1.0, norm_balance_clip_min: float=0.5, norm_balance_clip_max: float=1.5, novelty_boost_scale: float=0.5, hybrid_reference_decay: float=0.7, trust_region_enabled: bool=False, trust_region_coeff: float=1.0, guidance_scale: float=2.0, prior_action: Optional[torch.Tensor]=None, forcing_steps: int=0, return_cfg_debug: bool=False, save_cfg_vectors: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        B = first_frame.shape[0]
        reference_mode = str(cfg_reference_mode or 'runtime_uncond').strip().lower()
        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        uncond_language_embeddings = [emb.to(self.device).to(self.dtype) for emb in uncond_language_embeddings]
        if source_language_embeddings is not None:
            source_language_embeddings = [emb.to(self.device).to(self.dtype) for emb in source_language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)
        if uncond_first_frame is not None:
            uncond_first_frame = uncond_first_frame.to(self.device).to(self.dtype)
        if source_first_frame is not None:
            source_first_frame = source_first_frame.to(self.device).to(self.dtype)
        use_counterfactual = reference_mode in {'dual_residual_counterfactual', 'event_triggered_source_full', 'conflict_aware_event_triggered', 'conflict_aware_sparse_source_repulsion', 'alignment_gated_source_repulsion', 'semantic_metric_source_repulsion', 'dimension_masked_source_repulsion', 'source_memory_trace_cfg', 'hysteretic_event_cfg', 'soft_sign_conflict_cfg', 'norm_balanced_cfg', 'target_novelty_boost_cfg', 'hybrid_reference_takeover_cfg', 'novelty_hybrid_takeover_cfg', 'novelty_conflict_mask_cfg'} and source_language_embeddings is not None and (source_vlm_inputs is not None)
        resolved_lambda_plus = float(lambda_plus if lambda_plus is not None else 0.0)
        resolved_lambda_minus = float(lambda_minus if lambda_minus is not None else 0.0)
        resolved_event_gate_beta = float(event_gate_beta if event_gate_beta is not None else 0.0)
        resolved_event_gate_threshold = float(event_gate_threshold if event_gate_threshold is not None else 0.0)
        resolved_event_gate_stop_threshold = float(event_gate_stop_threshold if event_gate_stop_threshold is not None else 0.0)
        resolved_conflict_cos_threshold = float(conflict_cos_threshold if conflict_cos_threshold is not None else 0.0)
        resolved_conflict_mode = str(conflict_mode or 'negative_parallel_only').strip().lower()
        resolved_gate_mode = str(gate_mode or '').strip().lower()
        resolved_gate_on_threshold = float(gate_on_threshold if gate_on_threshold is not None else resolved_event_gate_threshold)
        resolved_gate_off_threshold = float(gate_off_threshold if gate_off_threshold is not None else resolved_event_gate_stop_threshold)
        resolved_gate_ema_decay = float(gate_ema_decay if gate_ema_decay is not None else 0.0)
        resolved_initial_gate_state = bool(initial_gate_state)
        resolved_initial_gate_ema_value = float(initial_gate_ema_value if initial_gate_ema_value is not None else 0.0)
        resolved_source_memory_decay = float(source_memory_decay if source_memory_decay is not None else 0.0)
        resolved_source_memory_mode = str(source_memory_mode or 'off').strip().lower()
        resolved_soft_mask_temperature = float(soft_mask_temperature if soft_mask_temperature is not None else 0.0)
        resolved_soft_mask_dominance_eta = float(soft_mask_dominance_eta if soft_mask_dominance_eta is not None else 0.0)
        resolved_norm_balance_clip_min = float(norm_balance_clip_min if norm_balance_clip_min is not None else 0.0)
        resolved_norm_balance_clip_max = float(norm_balance_clip_max if norm_balance_clip_max is not None else 0.0)
        resolved_novelty_boost_scale = float(novelty_boost_scale if novelty_boost_scale is not None else 0.0)
        resolved_hybrid_reference_decay = float(hybrid_reference_decay if hybrid_reference_decay is not None else 0.0)
        resolved_trust_region_enabled = bool(trust_region_enabled)
        resolved_trust_region_coeff = float(trust_region_coeff if trust_region_coeff is not None else 0.0)

        def _samplewise_rms(tensor: torch.Tensor) -> torch.Tensor:
            flat = tensor.reshape(tensor.shape[0], -1)
            denom = max(flat.shape[1], 1) ** 0.5
            return torch.linalg.vector_norm(flat, dim=1) / denom

        def _event_gate(target_residual: torch.Tensor, source_residual: torch.Tensor) -> torch.Tensor:
            disagreement = _samplewise_rms(target_residual - source_residual)
            return (torch.sigmoid(resolved_event_gate_beta * (disagreement - resolved_event_gate_threshold)), disagreement)

        def _conflict_project(source_residual: torch.Tensor, target_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            source_flat = source_residual.reshape(source_residual.shape[0], -1)
            target_flat = target_residual.reshape(target_residual.shape[0], -1)
            target_norm = torch.linalg.vector_norm(target_flat, dim=1, keepdim=True)
            source_norm = torch.linalg.vector_norm(source_flat, dim=1, keepdim=True)
            target_unit = target_flat / target_norm.clamp_min(1e-06)
            cosine = (source_flat * target_unit).sum(dim=1, keepdim=True) / source_norm.clamp_min(1e-06)
            conflict_mask = (cosine < resolved_conflict_cos_threshold).to(source_flat.dtype)
            if resolved_conflict_mode == 'negative_parallel_only':
                conflict_coeff = torch.clamp(-(source_flat * target_unit).sum(dim=1, keepdim=True), min=0.0)
                conflict_flat = conflict_mask * conflict_coeff * -target_unit
            else:
                conflict_flat = conflict_mask * source_flat
            conflict = conflict_flat.reshape_as(source_residual)
            return (conflict, cosine.squeeze(1), conflict_mask.squeeze(1))

        def _samplewise_cosine(lhs: Optional[torch.Tensor], rhs: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if lhs is None or rhs is None:
                return None
            lhs_flat = lhs.reshape(lhs.shape[0], -1)
            rhs_flat = rhs.reshape(rhs.shape[0], -1)
            lhs_norm = torch.linalg.vector_norm(lhs_flat, dim=1)
            rhs_norm = torch.linalg.vector_norm(rhs_flat, dim=1)
            denom = (lhs_norm * rhs_norm).clamp_min(1e-06)
            return (lhs_flat * rhs_flat).sum(dim=1) / denom

        def _alignment_gate_repulsion(source_residual: torch.Tensor, target_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            cosine = _samplewise_cosine(source_residual, target_residual)
            if cosine is None:
                raise ValueError('alignment-gated CFG requires both source and target residuals')
            gate = torch.clamp(-cosine, min=0.0)
            gate_shape = [source_residual.shape[0]] + [1] * (source_residual.ndim - 1)
            repulsion = gate.reshape(gate_shape) * source_residual
            return (repulsion, cosine, gate)

        def _pool_language_embeddings(embeddings: Optional[list[torch.Tensor]]) -> Optional[torch.Tensor]:
            if embeddings is None:
                return None
            pooled = []
            for emb in embeddings:
                if emb.ndim == 3 and emb.shape[0] == 1:
                    emb = emb.squeeze(0)
                if emb.ndim != 2:
                    raise ValueError(f'Expected per-sample language embedding with rank 2, got shape {tuple(emb.shape)}')
                pooled.append(emb.mean(dim=0))
            return torch.stack(pooled, dim=0)

        def _semantic_conflict_weight(target_embeddings: Optional[list[torch.Tensor]], source_embeddings: Optional[list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
            target_pooled = _pool_language_embeddings(target_embeddings)
            source_pooled = _pool_language_embeddings(source_embeddings)
            cosine = _samplewise_cosine(target_pooled, source_pooled)
            if cosine is None:
                raise ValueError('semantic-metric CFG requires both source and target condition embeddings')
            weight = torch.clamp(1.0 - cosine, min=0.0)
            return (weight, cosine)

        def _dimension_masked_repulsion(source_residual: torch.Tensor, target_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            sign_conflict = (source_residual * target_residual < 0).to(source_residual.dtype)
            denom = (source_residual.abs() + target_residual.abs()).clamp_min(1e-06)
            softness = (source_residual - target_residual).abs() / denom
            mask = sign_conflict * softness
            reduce_dims = tuple(range(1, mask.ndim))
            mask_fraction = sign_conflict.float().mean(dim=reduce_dims)
            mask_mean = mask.float().mean(dim=reduce_dims)
            return (mask * source_residual, mask_fraction, mask_mean)

        def _soft_sign_conflict_repulsion(source_residual: torch.Tensor, target_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            sign_weight = torch.sigmoid(-resolved_soft_mask_temperature * (source_residual * target_residual))
            dominance = torch.sigmoid(resolved_soft_mask_temperature * (source_residual.abs() - resolved_soft_mask_dominance_eta * target_residual.abs()))
            mask = sign_weight * dominance
            reduce_dims = tuple(range(1, mask.ndim))
            mask_fraction = (sign_weight > 0.5).float().mean(dim=reduce_dims)
            mask_mean = mask.float().mean(dim=reduce_dims)
            return (mask * source_residual, mask_fraction, mask_mean)

        def _orthogonal_component(target_residual: torch.Tensor, source_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            target_flat = target_residual.reshape(target_residual.shape[0], -1)
            source_flat = source_residual.reshape(source_residual.shape[0], -1)
            source_norm = torch.linalg.vector_norm(source_flat, dim=1, keepdim=True).clamp_min(1e-06)
            source_unit = source_flat / source_norm
            parallel_coeff = (target_flat * source_unit).sum(dim=1, keepdim=True)
            parallel = parallel_coeff * source_unit
            orthogonal = target_flat - parallel
            return (orthogonal.reshape_as(target_residual), parallel.reshape_as(target_residual))

        def _samplewise_gain(target_residual: torch.Tensor, repulsion: torch.Tensor) -> torch.Tensor:
            target_rms = _samplewise_rms(target_residual)
            repulsion_rms = _samplewise_rms(repulsion)
            ratio = target_rms / repulsion_rms.clamp_min(1e-06)
            clipped = torch.clamp(ratio, min=resolved_norm_balance_clip_min, max=resolved_norm_balance_clip_max)
            return resolved_lambda_minus * clipped

        def _apply_trust_region(video_velocity: torch.Tensor, action_velocity: torch.Tensor, video_reference: torch.Tensor, action_reference: torch.Tensor, action_target_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if not resolved_trust_region_enabled:
                ones = torch.ones(action_velocity.shape[0], device=action_velocity.device, dtype=action_velocity.dtype)
                return (video_velocity, action_velocity, ones)
            delta_action = action_velocity - action_reference
            delta_video = video_velocity - video_reference
            target_rms = _samplewise_rms(action_target_residual)
            delta_rms = _samplewise_rms(delta_action).clamp_min(1e-06)
            scale = torch.minimum(torch.ones_like(target_rms), resolved_trust_region_coeff * target_rms / delta_rms)
            action_shape = [delta_action.shape[0]] + [1] * (delta_action.ndim - 1)
            video_shape = [delta_video.shape[0]] + [1] * (delta_video.ndim - 1)
            action_velocity = action_reference + scale.view(action_shape) * delta_action
            video_velocity = video_reference + scale.view(video_shape) * delta_video
            return (video_velocity, action_velocity, scale)

        def _tensor_payload(tensor: Optional[torch.Tensor]) -> Optional[dict[str, Any]]:
            if tensor is None:
                return None
            tensor_cpu = tensor.detach().float().cpu()
            flat = tensor_cpu.reshape(tensor_cpu.shape[0], -1)
            payload = {'shape': [int(dim) for dim in tensor_cpu.shape], 'rms_mean': float(_samplewise_rms(tensor).detach().float().cpu().mean().item()), 'abs_max': float(tensor_cpu.abs().max().item())}
            if save_cfg_vectors:
                payload['vector'] = flat[0].tolist() if flat.shape[0] == 1 else flat.tolist()
            return payload
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))
            if uncond_first_frame is not None:
                uncond_first_frame_norm = (uncond_first_frame * 2.0 - 1.0).unsqueeze(2)
                uncond_condition_frame_latent = self.video_model.encode_video(uncond_first_frame_norm.to(self.dtype))
            else:
                uncond_condition_frame_latent = condition_frame_latent
            if use_counterfactual:
                if source_first_frame is not None:
                    source_first_frame_norm = (source_first_frame * 2.0 - 1.0).unsqueeze(2)
                    source_condition_frame_latent = self.video_model.encode_video(source_first_frame_norm.to(self.dtype))
                else:
                    source_condition_frame_latent = condition_frame_latent
            else:
                source_condition_frame_latent = None
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)
        if return_cfg_debug:
            video_latent_cond_track = video_latent.clone()
            action_latent_cond_track = action_latent.clone()
            video_latent_uncond_track = video_latent.clone()
            video_latent_uncond_track[:, :, 0:1] = uncond_condition_frame_latent
            action_latent_uncond_track = action_latent.clone()
            video_latent_source_track = None
            action_latent_source_track = None
            if use_counterfactual:
                video_latent_source_track = video_latent.clone()
                if source_condition_frame_latent is not None:
                    video_latent_source_track[:, :, 0:1] = source_condition_frame_latent
                action_latent_source_track = action_latent.clone()
        processed_t5_cond = self.video_module.preprocess_t5_embeddings(language_embeddings)
        processed_t5_uncond = self.video_module.preprocess_t5_embeddings(uncond_language_embeddings)
        processed_t5_source = self.video_module.preprocess_t5_embeddings(source_language_embeddings) if use_counterfactual else None
        semantic_conflict_weight = None
        semantic_condition_cosine = None
        if use_counterfactual and reference_mode == 'semantic_metric_source_repulsion':
            semantic_conflict_weight, semantic_condition_cosine = _semantic_conflict_weight(language_embeddings, source_language_embeddings)
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        last_event_gate = None
        last_disagreement = None
        last_conflict_cosine = None
        last_conflict_fraction = None
        last_alignment_gate = None
        last_semantic_conflict_weight = None
        last_semantic_condition_cosine = None
        last_dimension_mask_fraction = None
        last_dimension_mask_mean = None
        last_gate_state = bool(resolved_initial_gate_state)
        last_gate_ema_value = float(resolved_initial_gate_ema_value)
        last_memory_norm = None
        last_memory_target_cosine = None
        last_soft_mask_fraction = None
        last_soft_mask_mean = None
        last_norm_balance_gain = None
        last_trust_region_scale = None
        last_hybrid_reference_mu = None
        last_takeover_gap = None
        action_source_memory = None
        video_source_memory = None
        denoise_step_payloads: list[dict[str, Any]] = []
        for i in range(num_inference_steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - t
            sigma_k = float(t.detach().float().cpu().item())
            alpha_k = float(1.0 - sigma_k)
            t_embed_scalar = float((t * 1000).detach().float().cpu().item())
            action_latent_k = action_latent.detach().clone()
            video_vel_cond, action_vel_cond = self._single_denoise_step(video_latent, action_latent, state, t, processed_t5_cond, vlm_inputs, B)
            if uncond_first_frame is not None:
                video_latent_uncond_eval = video_latent.clone()
                video_latent_uncond_eval[:, :, 0:1] = uncond_condition_frame_latent
            else:
                video_latent_uncond_eval = video_latent
            video_vel_uncond, action_vel_uncond = self._single_denoise_step(video_latent_uncond_eval, action_latent, state, t, processed_t5_uncond, uncond_vlm_inputs, B)
            if use_counterfactual:
                if source_first_frame is not None:
                    video_latent_source_eval = video_latent.clone()
                    video_latent_source_eval[:, :, 0:1] = source_condition_frame_latent
                else:
                    video_latent_source_eval = video_latent
                video_vel_source, action_vel_source = self._single_denoise_step(video_latent_source_eval, action_latent, state, t, processed_t5_source, source_vlm_inputs, B)
                video_reference = video_vel_uncond
                action_reference = action_vel_uncond
                step_hybrid_reference_mu = None
                controller_gain = float(guidance_scale) if reference_mode in {'source_memory_trace_cfg', 'hysteretic_event_cfg', 'soft_sign_conflict_cfg', 'norm_balanced_cfg', 'target_novelty_boost_cfg', 'hybrid_reference_takeover_cfg', 'novelty_hybrid_takeover_cfg', 'novelty_conflict_mask_cfg'} else 1.0
                if reference_mode in {'hybrid_reference_takeover_cfg', 'novelty_hybrid_takeover_cfg'}:
                    step_hybrid_reference_mu = float(torch.exp(torch.tensor(-resolved_hybrid_reference_decay * i)).item())
                    video_reference = (1.0 - step_hybrid_reference_mu) * video_vel_uncond + step_hybrid_reference_mu * video_vel_source
                    action_reference = (1.0 - step_hybrid_reference_mu) * action_vel_uncond + step_hybrid_reference_mu * action_vel_source
                video_target_residual = video_vel_cond - video_reference
                action_target_residual = action_vel_cond - action_reference
                video_source_residual = video_vel_source - video_reference
                action_source_residual = action_vel_source - action_reference
                step_event_gate_value = None
                step_disagreement = None
                step_conflict_cosine = None
                step_conflict_fraction = None
                step_alignment_gate = None
                step_semantic_conflict_weight = None
                step_semantic_condition_cosine = None
                step_dimension_mask_fraction = None
                step_dimension_mask_mean = None
                step_gate_state = last_gate_state
                step_gate_ema_value = last_gate_ema_value
                step_memory_norm = None
                step_memory_target_cosine = None
                step_soft_mask_fraction = None
                step_soft_mask_mean = None
                step_norm_balance_gain = None
                step_trust_region_scale = None
                video_repulsion = video_source_residual
                action_repulsion = action_source_residual
                video_boost = torch.zeros_like(video_target_residual)
                action_boost = torch.zeros_like(action_target_residual)
                repulsion_gain = controller_gain * resolved_lambda_minus
                if reference_mode == 'event_triggered_source_full':
                    event_gate, disagreement = _event_gate(action_target_residual, action_source_residual)
                    video_repulsion = event_gate.view(B, 1, 1, 1, 1) * video_source_residual
                    action_repulsion = event_gate.view(B, 1, 1) * action_source_residual
                    step_event_gate_value = float(event_gate.max().item())
                    step_disagreement = float(disagreement.max().item())
                elif reference_mode == 'hysteretic_event_cfg':
                    raw_event_gate, disagreement = _event_gate(action_target_residual, action_source_residual)
                    prev_gate_ema = torch.full_like(raw_event_gate, last_gate_ema_value)
                    gate_signal = resolved_gate_ema_decay * prev_gate_ema + (1.0 - resolved_gate_ema_decay) * raw_event_gate if resolved_gate_ema_decay > 0.0 else raw_event_gate
                    prev_state = torch.full_like(gate_signal, 1.0 if last_gate_state else 0.0)
                    gate_state = torch.logical_or(gate_signal > resolved_gate_on_threshold, torch.logical_and(prev_state > 0.5, gate_signal > resolved_gate_off_threshold))
                    event_gate = gate_state.to(action_target_residual.dtype)
                    video_repulsion = event_gate.view(B, 1, 1, 1, 1) * video_source_residual
                    action_repulsion = event_gate.view(B, 1, 1) * action_source_residual
                    step_event_gate_value = float(gate_signal.max().item())
                    step_disagreement = float(disagreement.max().item())
                    step_gate_state = bool(gate_state.max().item())
                    step_gate_ema_value = float(gate_signal.max().item())
                elif reference_mode == 'conflict_aware_event_triggered':
                    event_gate, disagreement = _event_gate(action_target_residual, action_source_residual)
                    video_conflict, video_cosine, video_mask = _conflict_project(video_source_residual, video_target_residual)
                    action_conflict, action_cosine, action_mask = _conflict_project(action_source_residual, action_target_residual)
                    video_repulsion = event_gate.view(B, 1, 1, 1, 1) * video_conflict
                    action_repulsion = event_gate.view(B, 1, 1) * action_conflict
                    step_event_gate_value = float(event_gate.max().item())
                    step_disagreement = float(disagreement.max().item())
                    step_conflict_cosine = float(action_cosine.min().item())
                    step_conflict_fraction = float(action_mask.float().mean().item())
                elif reference_mode == 'conflict_aware_sparse_source_repulsion':
                    video_conflict, video_cosine, video_mask = _conflict_project(video_source_residual, video_target_residual)
                    action_conflict, action_cosine, action_mask = _conflict_project(action_source_residual, action_target_residual)
                    video_repulsion = video_conflict
                    action_repulsion = action_conflict
                    step_conflict_cosine = float(action_cosine.min().item())
                    step_conflict_fraction = float(action_mask.float().mean().item())
                elif reference_mode == 'alignment_gated_source_repulsion':
                    video_repulsion, video_cosine, video_gate = _alignment_gate_repulsion(video_source_residual, video_target_residual)
                    action_repulsion, action_cosine, action_gate = _alignment_gate_repulsion(action_source_residual, action_target_residual)
                    step_conflict_cosine = float(action_cosine.mean().item())
                    step_alignment_gate = float(action_gate.mean().item())
                elif reference_mode == 'semantic_metric_source_repulsion':
                    if semantic_conflict_weight is None or semantic_condition_cosine is None:
                        raise ValueError('semantic-metric CFG weight was not initialized')
                    video_repulsion = semantic_conflict_weight.view(B, 1, 1, 1, 1) * video_source_residual
                    action_repulsion = semantic_conflict_weight.view(B, 1, 1) * action_source_residual
                    step_semantic_conflict_weight = float(semantic_conflict_weight.mean().item())
                    step_semantic_condition_cosine = float(semantic_condition_cosine.mean().item())
                elif reference_mode == 'dimension_masked_source_repulsion':
                    video_repulsion, video_mask_fraction, video_mask_mean = _dimension_masked_repulsion(video_source_residual, video_target_residual)
                    action_repulsion, action_mask_fraction, action_mask_mean = _dimension_masked_repulsion(action_source_residual, action_target_residual)
                    action_cosine = _samplewise_cosine(action_source_residual, action_target_residual)
                    if action_cosine is not None:
                        step_conflict_cosine = float(action_cosine.mean().item())
                    step_conflict_fraction = float(action_mask_fraction.mean().item())
                    step_dimension_mask_fraction = float(action_mask_fraction.mean().item())
                    step_dimension_mask_mean = float(action_mask_mean.mean().item())
                elif reference_mode == 'source_memory_trace_cfg':
                    if action_source_memory is None:
                        action_source_memory = action_source_residual
                        video_source_memory = video_source_residual
                    else:
                        action_source_memory = resolved_source_memory_decay * action_source_memory + (1.0 - resolved_source_memory_decay) * action_source_residual
                        video_source_memory = resolved_source_memory_decay * video_source_memory + (1.0 - resolved_source_memory_decay) * video_source_residual
                    action_repulsion = action_source_memory
                    video_repulsion = video_source_memory
                    step_memory_norm = float(_samplewise_rms(action_source_memory).mean().item())
                    memory_target_cosine = _samplewise_cosine(action_source_memory, action_target_residual)
                    if memory_target_cosine is not None:
                        step_memory_target_cosine = float(memory_target_cosine.mean().item())
                elif reference_mode == 'soft_sign_conflict_cfg':
                    video_repulsion, video_mask_fraction, video_mask_mean = _soft_sign_conflict_repulsion(video_source_residual, video_target_residual)
                    action_repulsion, action_mask_fraction, action_mask_mean = _soft_sign_conflict_repulsion(action_source_residual, action_target_residual)
                    action_cosine = _samplewise_cosine(action_source_residual, action_target_residual)
                    if action_cosine is not None:
                        step_conflict_cosine = float(action_cosine.mean().item())
                    step_conflict_fraction = float(action_mask_fraction.mean().item())
                    step_soft_mask_fraction = float(action_mask_fraction.mean().item())
                    step_soft_mask_mean = float(action_mask_mean.mean().item())
                elif reference_mode == 'novelty_conflict_mask_cfg':
                    video_repulsion, video_mask_fraction, video_mask_mean = _soft_sign_conflict_repulsion(video_source_residual, video_target_residual)
                    action_repulsion, action_mask_fraction, action_mask_mean = _soft_sign_conflict_repulsion(action_source_residual, action_target_residual)
                    video_novelty, _ = _orthogonal_component(video_target_residual, video_source_residual)
                    action_novelty, _ = _orthogonal_component(action_target_residual, action_source_residual)
                    video_boost = resolved_novelty_boost_scale * video_novelty
                    action_boost = resolved_novelty_boost_scale * action_novelty
                    action_cosine = _samplewise_cosine(action_source_residual, action_target_residual)
                    if action_cosine is not None:
                        step_conflict_cosine = float(action_cosine.mean().item())
                    step_conflict_fraction = float(action_mask_fraction.mean().item())
                    step_soft_mask_fraction = float(action_mask_fraction.mean().item())
                    step_soft_mask_mean = float(action_mask_mean.mean().item())
                elif reference_mode == 'norm_balanced_cfg':
                    repulsion_gain = controller_gain * _samplewise_gain(action_target_residual, action_repulsion)
                    step_norm_balance_gain = float(repulsion_gain.mean().item())
                elif reference_mode in {'target_novelty_boost_cfg', 'novelty_hybrid_takeover_cfg'}:
                    video_novelty, _ = _orthogonal_component(video_target_residual, video_source_residual)
                    action_novelty, _ = _orthogonal_component(action_target_residual, action_source_residual)
                    video_boost = resolved_novelty_boost_scale * video_novelty
                    action_boost = resolved_novelty_boost_scale * action_novelty
                if isinstance(repulsion_gain, torch.Tensor):
                    action_gain = repulsion_gain.view(B, 1, 1)
                    video_gain = repulsion_gain.view(B, 1, 1, 1, 1)
                else:
                    action_gain = resolved_lambda_minus
                    video_gain = resolved_lambda_minus
                base_video = video_reference if reference_mode in {'hybrid_reference_takeover_cfg', 'novelty_hybrid_takeover_cfg'} else video_vel_cond
                base_action = action_reference if reference_mode in {'hybrid_reference_takeover_cfg', 'novelty_hybrid_takeover_cfg'} else action_vel_cond
                video_velocity = base_video + controller_gain * resolved_lambda_plus * video_target_residual + video_boost - video_gain * video_repulsion
                action_velocity = base_action + controller_gain * resolved_lambda_plus * action_target_residual + action_boost - action_gain * action_repulsion
                video_velocity, action_velocity, trust_region_scale = _apply_trust_region(video_velocity, action_velocity, video_reference, action_reference, action_target_residual)
                if resolved_trust_region_enabled:
                    step_trust_region_scale = float(trust_region_scale.mean().item())
                last_event_gate = step_event_gate_value
                last_disagreement = step_disagreement
                last_conflict_cosine = step_conflict_cosine
                last_conflict_fraction = step_conflict_fraction
                last_alignment_gate = step_alignment_gate
                last_semantic_conflict_weight = step_semantic_conflict_weight
                last_semantic_condition_cosine = step_semantic_condition_cosine
                last_dimension_mask_fraction = step_dimension_mask_fraction
                last_dimension_mask_mean = step_dimension_mask_mean
                last_gate_state = step_gate_state
                last_gate_ema_value = step_gate_ema_value
                last_memory_norm = step_memory_norm
                last_memory_target_cosine = step_memory_target_cosine
                last_soft_mask_fraction = step_soft_mask_fraction
                last_soft_mask_mean = step_soft_mask_mean
                last_norm_balance_gain = step_norm_balance_gain
                last_trust_region_scale = step_trust_region_scale
                last_hybrid_reference_mu = step_hybrid_reference_mu
            else:
                video_velocity = video_vel_uncond + guidance_scale * (video_vel_cond - video_vel_uncond)
                action_velocity = action_vel_uncond + guidance_scale * (action_vel_cond - action_vel_uncond)
                action_source_residual = None
                action_target_residual = action_vel_cond - action_vel_uncond
                last_event_gate = None
                last_disagreement = None
                last_conflict_cosine = None
                last_conflict_fraction = None
                last_alignment_gate = None
                last_semantic_conflict_weight = None
                last_semantic_condition_cosine = None
                last_dimension_mask_fraction = None
                last_dimension_mask_mean = None
                last_gate_state = False
                last_gate_ema_value = 0.0
                last_memory_norm = None
                last_memory_target_cosine = None
                last_soft_mask_fraction = None
                last_soft_mask_mean = None
                last_norm_balance_gain = None
                last_trust_region_scale = None
                last_hybrid_reference_mu = None
            if return_cfg_debug:
                action_guidance_reference = action_reference if use_counterfactual else action_vel_uncond
                guided_action_residual = action_velocity - action_guidance_reference
                target_source_cosine = _samplewise_cosine(action_target_residual, action_source_residual)
                guided_target_cosine = _samplewise_cosine(guided_action_residual, action_target_residual)
                guided_source_cosine = _samplewise_cosine(guided_action_residual, action_source_residual)
                takeover_gap = None
                if guided_target_cosine is not None and guided_source_cosine is not None:
                    takeover_gap = float((guided_target_cosine.detach().float().cpu().mean() - guided_source_cosine.detach().float().cpu().mean()).item())
                    last_takeover_gap = takeover_gap
                action_x0_target = action_latent_k - sigma_k * action_vel_cond
                action_x0_null = action_latent_k - sigma_k * action_vel_uncond
                action_x0_guided = action_latent_k - sigma_k * action_velocity
                step_payload: dict[str, Any] = {'denoise_step_index': int(i + 1), 't': float(t.detach().float().cpu().item()), 'dt': float(dt.detach().float().cpu().item()), 'sigma': sigma_k, 'alpha': alpha_k, 'scheduler_family': 'manual_euler_flow_matching', 'scheduler_coeffs': {'clean_coeff': alpha_k, 'noise_coeff': sigma_k, 'update_coeff': float(dt.detach().float().cpu().item()), 't_embed_scalar': t_embed_scalar}, 'prediction_type': 'flow_matching_velocity', 'cfg_reference_mode': reference_mode, 'action_x_k': _tensor_payload(action_latent_k), 'action_eps_target': _tensor_payload(action_vel_cond), 'action_eps_null': _tensor_payload(action_vel_uncond), 'action_eps_guided': _tensor_payload(action_velocity), 'action_x0_target': _tensor_payload(action_x0_target), 'action_x0_null': _tensor_payload(action_x0_null), 'action_x0_guided': _tensor_payload(action_x0_guided), 'action_target_residual': _tensor_payload(action_target_residual), 'action_guided_residual': _tensor_payload(guided_action_residual)}
                if action_source_residual is not None:
                    target_source_delta = action_target_residual - action_source_residual
                    action_x0_source = action_latent_k - sigma_k * action_vel_source
                    step_payload['action_eps_source'] = _tensor_payload(action_vel_source)
                    step_payload['action_x0_source'] = _tensor_payload(action_x0_source)
                    step_payload['action_source_residual'] = _tensor_payload(action_source_residual)
                    step_payload['target_source_delta'] = _tensor_payload(target_source_delta)
                if target_source_cosine is not None:
                    step_payload['target_source_cosine_mean'] = float(target_source_cosine.detach().float().cpu().mean().item())
                if guided_target_cosine is not None:
                    step_payload['guided_target_cosine_mean'] = float(guided_target_cosine.detach().float().cpu().mean().item())
                if guided_source_cosine is not None:
                    step_payload['guided_source_cosine_mean'] = float(guided_source_cosine.detach().float().cpu().mean().item())
                if takeover_gap is not None:
                    step_payload['takeover_gap'] = takeover_gap
                if last_event_gate is not None:
                    step_payload['event_gate_value'] = float(last_event_gate)
                if last_disagreement is not None:
                    step_payload['event_gate_disagreement_norm'] = float(last_disagreement)
                step_payload['gate_state'] = bool(last_gate_state)
                step_payload['gate_ema_value'] = float(last_gate_ema_value)
                if last_conflict_cosine is not None:
                    step_payload['conflict_cosine'] = float(last_conflict_cosine)
                if last_conflict_fraction is not None:
                    step_payload['conflict_fraction'] = float(last_conflict_fraction)
                if last_alignment_gate is not None:
                    step_payload['alignment_gate_value'] = float(last_alignment_gate)
                if last_semantic_conflict_weight is not None:
                    step_payload['semantic_conflict_weight'] = float(last_semantic_conflict_weight)
                if last_semantic_condition_cosine is not None:
                    step_payload['semantic_condition_cosine'] = float(last_semantic_condition_cosine)
                if last_dimension_mask_fraction is not None:
                    step_payload['dimension_mask_fraction'] = float(last_dimension_mask_fraction)
                if last_dimension_mask_mean is not None:
                    step_payload['dimension_mask_mean'] = float(last_dimension_mask_mean)
                if last_memory_norm is not None:
                    step_payload['memory_norm'] = float(last_memory_norm)
                if last_memory_target_cosine is not None:
                    step_payload['memory_target_cosine'] = float(last_memory_target_cosine)
                if last_soft_mask_fraction is not None:
                    step_payload['soft_mask_fraction'] = float(last_soft_mask_fraction)
                if last_soft_mask_mean is not None:
                    step_payload['soft_mask_mean'] = float(last_soft_mask_mean)
                if last_norm_balance_gain is not None:
                    step_payload['norm_balance_gain'] = float(last_norm_balance_gain)
                if last_trust_region_scale is not None:
                    step_payload['trust_region_scale'] = float(last_trust_region_scale)
                if last_hybrid_reference_mu is not None:
                    step_payload['hybrid_reference_mu'] = float(last_hybrid_reference_mu)
                denoise_step_payloads.append(step_payload)
            video_latent = video_latent + video_velocity * dt
            action_latent = action_latent + action_velocity * dt
            if return_cfg_debug:
                video_vel_cond_exact, action_vel_cond_exact = self._single_denoise_step(video_latent_cond_track, action_latent_cond_track, state, t, processed_t5_cond, vlm_inputs, B)
                video_vel_uncond_exact, action_vel_uncond_exact = self._single_denoise_step(video_latent_uncond_track, action_latent_uncond_track, state, t, processed_t5_uncond, uncond_vlm_inputs, B)
                video_latent_cond_track = video_latent_cond_track + video_vel_cond_exact * dt
                action_latent_cond_track = action_latent_cond_track + action_vel_cond_exact * dt
                video_latent_uncond_track = video_latent_uncond_track + video_vel_uncond_exact * dt
                action_latent_uncond_track = action_latent_uncond_track + action_vel_uncond_exact * dt
                if use_counterfactual:
                    video_vel_source_exact, action_vel_source_exact = self._single_denoise_step(video_latent_source_track, action_latent_source_track, state, t, processed_t5_source, source_vlm_inputs, B)
                    video_latent_source_track = video_latent_source_track + video_vel_source_exact * dt
                    action_latent_source_track = action_latent_source_track + action_vel_source_exact * dt
            forced = None
            if prior_action is not None and forcing_steps > 0:
                t_next = timesteps[i + 1]
                noise = torch.randn_like(prior_action[:, :forcing_steps, :])
                forced = (1 - t_next) * prior_action[:, :forcing_steps, :].to(self.dtype) + t_next * noise
                action_latent[:, :forcing_steps, :] = forced
                if return_cfg_debug:
                    action_latent_cond_track[:, :forcing_steps, :] = forced
                    action_latent_uncond_track[:, :forcing_steps, :] = forced
                    if use_counterfactual and action_latent_source_track is not None:
                        action_latent_source_track[:, :forcing_steps, :] = forced
            video_latent[:, :, 0:1] = condition_frame_latent
            if return_cfg_debug:
                video_latent_cond_track[:, :, 0:1] = condition_frame_latent
                video_latent_uncond_track[:, :, 0:1] = uncond_condition_frame_latent
                if use_counterfactual and video_latent_source_track is not None and (source_condition_frame_latent is not None):
                    video_latent_source_track[:, :, 0:1] = source_condition_frame_latent
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]
            predicted_frames = (predicted_frames + 1.0) / 2.0
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
        predicted_actions = action_latent.float()
        if return_cfg_debug:
            cfg_debug = {'predicted_actions_final': predicted_actions, 'predicted_actions_cond': action_latent_cond_track.float(), 'predicted_actions_uncond': action_latent_uncond_track.float(), 'predicted_actions_target': action_latent_cond_track.float(), 'predicted_actions_null': action_latent_uncond_track.float(), 'cfg_reference_mode': reference_mode, 'prediction_type': 'flow_matching_velocity', 'scheduler_family': 'manual_euler_flow_matching', 'lambda_plus': resolved_lambda_plus, 'lambda_minus': resolved_lambda_minus, 'event_gate_value': last_event_gate, 'event_gate_disagreement_norm': last_disagreement, 'conflict_cosine': last_conflict_cosine, 'conflict_fraction': last_conflict_fraction, 'alignment_gate_value': last_alignment_gate, 'semantic_conflict_weight': last_semantic_conflict_weight, 'semantic_condition_cosine': last_semantic_condition_cosine, 'dimension_mask_fraction': last_dimension_mask_fraction, 'dimension_mask_mean': last_dimension_mask_mean, 'gate_state_final': last_gate_state, 'gate_ema_value_final': last_gate_ema_value, 'memory_norm': last_memory_norm, 'memory_target_cosine': last_memory_target_cosine, 'soft_mask_fraction': last_soft_mask_fraction, 'soft_mask_mean': last_soft_mask_mean, 'norm_balance_gain': last_norm_balance_gain, 'trust_region_scale': last_trust_region_scale, 'hybrid_reference_mu': last_hybrid_reference_mu, 'takeover_gap': last_takeover_gap, 'denoise_step_payloads': denoise_step_payloads}
            if use_counterfactual and action_latent_source_track is not None:
                cfg_debug['predicted_actions_source'] = action_latent_source_track.float()
            return (predicted_frames, predicted_actions, cfg_debug)
        return (predicted_frames, predicted_actions)
