import torch
import torch.nn as nn
import numpy as np
import cv2
import importlib.util
from pathlib import Path
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1, encoding='utf-8')
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1, encoding='utf-8')
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'models'))
from models.motus import Motus, MotusConfig
BAK_ROOT = str((Path(__file__).parent / 'bak').resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)
from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding
logger = logging.getLogger(__name__)

class MotusPolicy:

    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, vlm_path: str, device: str='cuda', log_dir: Optional[str]=None, task_name: Optional[str]=None, progress_checkpoint_path: Optional[str]=None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        self.progress_checkpoint_path = progress_checkpoint_path
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        self.model = self._load_model()
        self.t5_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=device, checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'), tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'))
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None
        self.forcing_steps = 0
        self.forcing_prior_action = None
        self.cfg_enabled = False
        self.guidance_scale = 2.0
        self.cfg_schedule_name = 'constant'
        self.cfg_schedule_scales = []
        self.cfg_reference_mode = 'runtime_uncond'
        self.cfg_reference_label = 'runtime_uncond'
        self.cfg_reference_instruction = ''
        self.cfg_source_instruction = ''
        self.cfg_lambda_plus = 1.0
        self.cfg_lambda_minus = 0.0
        self.cfg_event_gate_beta = 10.0
        self.cfg_event_gate_threshold = 0.25
        self.cfg_event_gate_stop_threshold = 0.5
        self.cfg_conflict_cos_threshold = 0.0
        self.cfg_conflict_mode = 'negative_parallel_only'
        self.cfg_gate_mode = ''
        self.cfg_gate_on_threshold = 0.25
        self.cfg_gate_off_threshold = 0.15
        self.cfg_gate_ema_decay = 0.7
        self.cfg_gate_state = False
        self.cfg_gate_ema_value = 0.0
        self.cfg_gate_history = []
        self.cfg_source_memory_decay = 0.7
        self.cfg_source_memory_mode = 'off'
        self.cfg_soft_mask_temperature = 8.0
        self.cfg_soft_mask_dominance_eta = 1.0
        self.cfg_norm_balance_clip_min = 0.5
        self.cfg_norm_balance_clip_max = 1.5
        self.cfg_novelty_boost_scale = 0.5
        self.cfg_hybrid_reference_decay = 0.7
        self.cfg_trust_region_enabled = False
        self.cfg_trust_region_coeff = 1.0
        self.cfg_continuity_guard = 'off'
        self.cfg_switch_window = 0
        self.cfg_target_branch = 'target'
        self.cfg_null_branch = 'null'
        self.cfg_source_branch = 'source'
        self.cfg_projection_mode = 'off'
        self.cfg_projection_strength = 0.0
        self.cfg_projection_prefix_steps = 2
        self.cfg_activation_window_mode = ''
        self.uncond_instruction = ''
        self.uncond_visual_mode = 'clear'
        self.uncond_blur_kernel_size = 0
        self.uncond_blur_sigma = 0.0
        self.cfg_inference_count = 0
        self.cfg_max_inferences = 3
        self.current_instruction = ''
        self.switch_trace_enabled = False
        self.trace_stat_keys = ['mean_abs_all', 'std_all', 'mean_l2']
        self.switch_metadata = {}
        self.switch_cfg_scale_history = []
        self.switch_event_gate_history = []
        self.post_switch_action_trace = []
        self.cfg_whitebox_enabled = False
        self.cfg_whitebox_save_vectors = False
        self.cfg_whitebox_first_k = 5
        self.cfg_whitebox_zero_eps = 0.001
        self.cfg_whitebox_payload = {}
        self.last_executed_action = None
        self.last_action_chunk = None
        self.offline_trace_enabled = False
        self.offline_corpus_role = ''
        self.offline_trace_mode = 'off'
        self.offline_trace_window = 0
        self.offline_space_name = 'action_chunk'
        self.offline_export_tag = ''
        self.offline_episode_action_trace = []
        self.offline_episode_action_chunk_trace = []
        self.bridge_search_enabled = False
        self.bridge_inferences_remaining = 0
        self.bridge_candidate_scales = []
        self.bridge_blend_weights = []
        self.bridge_prefix_steps = 2
        self.bridge_projection_strength = 0.25
        self.bridge_target_weight = 1.0
        self.bridge_source_guard_weight = 0.3
        self.bridge_boundary_weight = 0.2
        self.bridge_uncertainty_weight = 0.1
        self.bridge_source_actions = None
        self.bridge_selection_history = []
        self._load_normalization_stats()
        save_images_env = os.environ.get('MOTUS_SAVE_IMAGES')
        if save_images_env is None:
            self.save_images = True
        else:
            self.save_images = str(save_images_env).strip().lower() not in {'0', 'false', 'no', 'off'}
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / 'logs')
        task_dir_name = task_name or os.environ.get('TASK_NAME') or 'default_task'
        self.save_dir = Path(base_log_dir) / 'images' / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.step_count = 0
        self.inference_times = []
        self.total_inference_time = 0.0
        self.inference_count = 0
        self.progress_enabled = False
        self.progress_head = None
        self.progress_feature_source = 'motus_visual'
        self.progress_hidden_state = None
        self.progress_scores = []
        self.latest_progress_score = None
        self.latest_action_progress_feature = None
        self.progress_switch_consumed = False
        self.progress_trigger_metadata = {}
        self.progress_dry_run_trigger_metadata = {}
        self.progress_trace_metadata = {}
        self.progress_trace_history = []
        if self.progress_checkpoint_path:
            self._load_progress_head(self.progress_checkpoint_path)
        logger.info('Motus Policy initialized successfully')

    def set_instruction(self, instruction: str):
        self.current_instruction = instruction
        logger.info(f'Instruction set: {instruction}')

    def prepare_instruction_switch(self, clear_action_cache: bool=True, enable_forcing: bool=False, forcing_steps: int=1, enable_cfg: bool=False, guidance_scale: float=2.0, cfg_max_inferences: Optional[int]=None, cfg_schedule_name: Optional[str]=None, cfg_schedule_scales: Optional[List[float]]=None, uncond_instruction: str='', cfg_reference_mode: str='runtime_uncond', cfg_reference_label: str='', cfg_reference_instruction: str='', lambda_plus: float=1.0, lambda_minus: float=0.0, cfg_event_gate_beta: float=10.0, cfg_event_gate_threshold: float=0.25, cfg_event_gate_stop_threshold: float=0.5, cfg_conflict_cos_threshold: float=0.0, cfg_conflict_mode: str='negative_parallel_only', cfg_gate_mode: str='', cfg_gate_on_threshold: float=0.25, cfg_gate_off_threshold: float=0.15, cfg_gate_ema_decay: float=0.7, cfg_source_memory_decay: float=0.7, cfg_source_memory_mode: str='off', cfg_soft_mask_temperature: float=8.0, cfg_soft_mask_dominance_eta: float=1.0, cfg_norm_balance_clip_min: float=0.5, cfg_norm_balance_clip_max: float=1.5, cfg_novelty_boost_scale: float=0.5, cfg_hybrid_reference_decay: float=0.7, cfg_trust_region_enabled: bool=False, cfg_trust_region_coeff: float=1.0, continuity_guard: str='off', switch_window: Optional[int]=None, target_branch: str='target', null_branch: str='null', source_branch: str='source', uncond_visual_mode: str='clear', uncond_blur_kernel_size: int=0, uncond_blur_sigma: float=0.0, cfg_projection_mode: str='off', cfg_projection_strength: float=0.0, cfg_projection_prefix_steps: int=2, cfg_activation_window_mode: str='', switch_trace_enabled: bool=False, trace_stats: Optional[List[str]]=None, cfg_whitebox_enabled: bool=False, cfg_whitebox_save_vectors: bool=False, cfg_whitebox_first_k: int=5, enable_bridge_search: bool=False, bridge_num_candidates: int=0, bridge_window_inferences: int=1, bridge_candidate_scales: Optional[List[float]]=None, bridge_blend_weights: Optional[List[float]]=None, bridge_prefix_steps: int=2, bridge_projection_strength: float=0.25, bridge_target_weight: float=1.0, bridge_source_guard_weight: float=0.3, bridge_boundary_weight: float=0.2, bridge_uncertainty_weight: float=0.1) -> Dict[str, Any]:
        cache_len_before = len(self.action_cache)
        prev_action_available = self.prev_action is not None
        cached_actions_before_switch = list(self.action_cache)
        source_actions = self._capture_bridge_source_actions(cached_actions_before_switch)
        if clear_action_cache:
            self.action_cache.clear()
        self.forcing_prior_action = None
        self.forcing_steps = 0
        self.cfg_enabled = False
        self.cfg_inference_count = 0
        self.switch_trace_enabled = bool(switch_trace_enabled)
        self.trace_stat_keys = [str(name) for name in trace_stats or ['mean_abs_all', 'std_all', 'mean_l2']]
        self.switch_cfg_scale_history = []
        self.switch_event_gate_history = []
        self.cfg_gate_history = []
        self.cfg_gate_state = False
        self.cfg_gate_ema_value = 0.0
        self.post_switch_action_trace = []
        self.cfg_whitebox_enabled = bool(cfg_whitebox_enabled)
        self.cfg_whitebox_save_vectors = bool(cfg_whitebox_save_vectors)
        self.cfg_whitebox_first_k = max(int(cfg_whitebox_first_k or 0), 0)
        self.cfg_whitebox_payload = {}
        if self.cfg_whitebox_enabled and (not enable_cfg):
            self.cfg_whitebox_payload['warning'] = 'cfg_whitebox_enabled_but_cfg_disabled'
        self.uncond_visual_mode = self._normalize_uncond_visual_mode(uncond_visual_mode)
        self.uncond_blur_kernel_size = self._normalize_blur_kernel_size(uncond_blur_kernel_size)
        self.uncond_blur_sigma = max(float(uncond_blur_sigma or 0.0), 0.0)
        self.bridge_search_enabled = False
        self.bridge_inferences_remaining = 0
        self.bridge_candidate_scales = []
        self.bridge_blend_weights = []
        self.bridge_prefix_steps = max(int(bridge_prefix_steps or 0), 0)
        self.bridge_projection_strength = max(float(bridge_projection_strength or 0.0), 0.0)
        self.bridge_target_weight = float(bridge_target_weight or 0.0)
        self.bridge_source_guard_weight = float(bridge_source_guard_weight or 0.0)
        self.bridge_boundary_weight = float(bridge_boundary_weight or 0.0)
        self.bridge_uncertainty_weight = float(bridge_uncertainty_weight or 0.0)
        self.bridge_source_actions = source_actions
        self.bridge_selection_history = []
        resolved_schedule = [float(scale) for scale in cfg_schedule_scales or []]
        resolved_max_inferences = int(cfg_max_inferences if cfg_max_inferences is not None else self.cfg_max_inferences)
        resolved_reference_mode = self._normalize_cfg_reference_mode(cfg_reference_mode)
        resolved_lambda_plus = float(lambda_plus or 0.0)
        resolved_lambda_minus = float(lambda_minus or 0.0)
        resolved_event_gate_beta = float(cfg_event_gate_beta or 0.0)
        resolved_event_gate_threshold = float(cfg_event_gate_threshold or 0.0)
        resolved_event_gate_stop_threshold = float(cfg_event_gate_stop_threshold or 0.0)
        resolved_conflict_cos_threshold = float(cfg_conflict_cos_threshold or 0.0)
        resolved_conflict_mode = str(cfg_conflict_mode or 'negative_parallel_only').strip().lower()
        resolved_gate_mode = str(cfg_gate_mode or '').strip().lower()
        resolved_gate_on_threshold = float(cfg_gate_on_threshold or 0.0)
        resolved_gate_off_threshold = float(cfg_gate_off_threshold or 0.0)
        resolved_gate_ema_decay = float(cfg_gate_ema_decay or 0.0)
        resolved_source_memory_decay = float(cfg_source_memory_decay or 0.0)
        resolved_source_memory_mode = str(cfg_source_memory_mode or 'off').strip().lower()
        resolved_soft_mask_temperature = float(cfg_soft_mask_temperature or 0.0)
        resolved_soft_mask_dominance_eta = float(cfg_soft_mask_dominance_eta or 0.0)
        resolved_norm_balance_clip_min = float(cfg_norm_balance_clip_min or 0.0)
        resolved_norm_balance_clip_max = float(cfg_norm_balance_clip_max or 0.0)
        resolved_novelty_boost_scale = float(cfg_novelty_boost_scale or 0.0)
        resolved_hybrid_reference_decay = float(cfg_hybrid_reference_decay or 0.0)
        resolved_trust_region_enabled = bool(cfg_trust_region_enabled)
        resolved_trust_region_coeff = float(cfg_trust_region_coeff or 0.0)
        resolved_continuity_guard = self._normalize_continuity_guard(continuity_guard)
        resolved_target_branch = str(target_branch or 'target')
        resolved_null_branch = str(null_branch or 'null')
        resolved_source_branch = str(source_branch or 'source')
        resolved_source_instruction = str(self.current_instruction or '')
        uncond_blur_active = bool(enable_cfg and self.uncond_visual_mode == 'blurred' and (self.uncond_blur_kernel_size > 1))
        forcing_active = False
        if enable_forcing and prev_action_available:
            prior = torch.from_numpy(self.prev_action).float().unsqueeze(0).unsqueeze(0)
            self.forcing_prior_action = prior.to(self.device)
            self.forcing_steps = forcing_steps
            forcing_active = True
        if enable_cfg:
            self.cfg_enabled = True
            if not resolved_schedule:
                resolved_schedule = [float(guidance_scale)] * max(resolved_max_inferences, 1)
            if resolved_max_inferences <= 0:
                resolved_max_inferences = len(resolved_schedule)
            self.guidance_scale = float(resolved_schedule[0])
            self.cfg_schedule_name = cfg_schedule_name or 'constant'
            self.cfg_schedule_scales = resolved_schedule
            self.cfg_reference_mode = resolved_reference_mode
            self.cfg_reference_label = str(cfg_reference_label or self.cfg_reference_mode)
            self.cfg_reference_instruction = str(cfg_reference_instruction or uncond_instruction or '')
            self.cfg_projection_mode = self._normalize_cfg_projection_mode(cfg_projection_mode)
            resolved_projection_strength = max(float(cfg_projection_strength or 0.0), 0.0)
            self.cfg_projection_strength = resolved_projection_strength if self.cfg_projection_mode == 'on' else 0.0
            self.cfg_projection_prefix_steps = max(int(cfg_projection_prefix_steps or 0), 0)
            self.cfg_activation_window_mode = str(cfg_activation_window_mode or self.cfg_schedule_name or '')
            self.uncond_instruction = self.cfg_reference_instruction
            self.cfg_inference_count = 0
            self.cfg_max_inferences = max(resolved_max_inferences, 1)
        else:
            self.guidance_scale = float(guidance_scale)
            self.cfg_schedule_name = cfg_schedule_name or 'none'
            self.cfg_schedule_scales = []
            self.cfg_reference_mode = resolved_reference_mode
            self.cfg_reference_label = str(cfg_reference_label or self.cfg_reference_mode)
            self.cfg_reference_instruction = str(cfg_reference_instruction or uncond_instruction or '')
            self.cfg_projection_mode = self._normalize_cfg_projection_mode(cfg_projection_mode)
            resolved_projection_strength = max(float(cfg_projection_strength or 0.0), 0.0)
            self.cfg_projection_strength = resolved_projection_strength if self.cfg_projection_mode == 'on' else 0.0
            self.cfg_projection_prefix_steps = max(int(cfg_projection_prefix_steps or 0), 0)
            self.cfg_activation_window_mode = str(cfg_activation_window_mode or self.cfg_schedule_name or '')
            self.uncond_instruction = self.cfg_reference_instruction
            self.cfg_max_inferences = 0
        self.cfg_source_instruction = resolved_source_instruction
        self.cfg_lambda_plus = resolved_lambda_plus
        self.cfg_lambda_minus = resolved_lambda_minus
        self.cfg_event_gate_beta = resolved_event_gate_beta
        self.cfg_event_gate_threshold = resolved_event_gate_threshold
        self.cfg_event_gate_stop_threshold = resolved_event_gate_stop_threshold
        self.cfg_conflict_cos_threshold = resolved_conflict_cos_threshold
        self.cfg_conflict_mode = resolved_conflict_mode
        self.cfg_gate_mode = resolved_gate_mode
        self.cfg_gate_on_threshold = resolved_gate_on_threshold
        self.cfg_gate_off_threshold = resolved_gate_off_threshold
        self.cfg_gate_ema_decay = resolved_gate_ema_decay
        self.cfg_source_memory_decay = resolved_source_memory_decay
        self.cfg_source_memory_mode = resolved_source_memory_mode
        self.cfg_soft_mask_temperature = resolved_soft_mask_temperature
        self.cfg_soft_mask_dominance_eta = resolved_soft_mask_dominance_eta
        self.cfg_norm_balance_clip_min = resolved_norm_balance_clip_min
        self.cfg_norm_balance_clip_max = resolved_norm_balance_clip_max
        self.cfg_novelty_boost_scale = resolved_novelty_boost_scale
        self.cfg_hybrid_reference_decay = resolved_hybrid_reference_decay
        self.cfg_trust_region_enabled = resolved_trust_region_enabled
        self.cfg_trust_region_coeff = resolved_trust_region_coeff
        self.cfg_continuity_guard = resolved_continuity_guard
        self.cfg_switch_window = max(int(switch_window if switch_window is not None else resolved_max_inferences), 0)
        self.cfg_target_branch = resolved_target_branch
        self.cfg_null_branch = resolved_null_branch
        self.cfg_source_branch = resolved_source_branch
        resolved_bridge_candidate_scales = [float(scale) for scale in bridge_candidate_scales or []]
        if not resolved_bridge_candidate_scales:
            if resolved_schedule:
                resolved_bridge_candidate_scales = list(resolved_schedule)
            elif enable_cfg:
                resolved_bridge_candidate_scales = [float(guidance_scale)]
        requested_bridge_num_candidates = max(int(bridge_num_candidates or 0), 0)
        if requested_bridge_num_candidates > 0 and resolved_bridge_candidate_scales:
            if len(resolved_bridge_candidate_scales) < requested_bridge_num_candidates:
                resolved_bridge_candidate_scales.extend([resolved_bridge_candidate_scales[-1]] * (requested_bridge_num_candidates - len(resolved_bridge_candidate_scales)))
            else:
                resolved_bridge_candidate_scales = resolved_bridge_candidate_scales[:requested_bridge_num_candidates]
        resolved_bridge_blend_weights = [float(weight) for weight in bridge_blend_weights or []]
        if resolved_bridge_candidate_scales and (not resolved_bridge_blend_weights):
            if len(resolved_bridge_candidate_scales) == 1:
                resolved_bridge_blend_weights = [0.35]
            else:
                denom = max(len(resolved_bridge_candidate_scales) - 1, 1)
                resolved_bridge_blend_weights = [0.65 * (1.0 - float(index) / float(denom)) for index in range(len(resolved_bridge_candidate_scales))]
        if resolved_bridge_candidate_scales and resolved_bridge_blend_weights:
            if len(resolved_bridge_blend_weights) < len(resolved_bridge_candidate_scales):
                resolved_bridge_blend_weights.extend([resolved_bridge_blend_weights[-1]] * (len(resolved_bridge_candidate_scales) - len(resolved_bridge_blend_weights)))
            else:
                resolved_bridge_blend_weights = resolved_bridge_blend_weights[:len(resolved_bridge_candidate_scales)]
        bridge_requested = bool(enable_bridge_search)
        bridge_warning = None
        if bridge_requested and (not enable_cfg):
            bridge_warning = 'iti2_bridge_search_requires_cfg_enabled'
        elif bridge_requested and (not resolved_bridge_candidate_scales):
            bridge_warning = 'iti2_bridge_search_requires_candidate_scales'
        self.bridge_search_enabled = bool(bridge_requested and enable_cfg and resolved_bridge_candidate_scales and (max(int(bridge_window_inferences or 0), 0) > 0))
        if bridge_requested and self.bridge_source_actions is None:
            bridge_warning = 'iti2_bridge_search_missing_source_actions'
        self.bridge_inferences_remaining = max(int(bridge_window_inferences or 0), 0) if self.bridge_search_enabled else 0
        self.bridge_candidate_scales = list(resolved_bridge_candidate_scales) if self.bridge_search_enabled else []
        self.bridge_blend_weights = list(resolved_bridge_blend_weights) if self.bridge_search_enabled else []
        self.switch_metadata = {'cache_len_before_switch': cache_len_before, 'prev_action_available': prev_action_available, 'cache_cleared': bool(clear_action_cache), 'forcing_requested': bool(enable_forcing), 'forcing_enabled': forcing_active, 'cfg_enabled': bool(enable_cfg), 'guidance_scale': self.guidance_scale if enable_cfg else None, 'cfg_max_inferences': self.cfg_max_inferences, 'cfg_schedule_name': self.cfg_schedule_name, 'cfg_schedule_scales': list(self.cfg_schedule_scales), 'cfg_reference_mode': self.cfg_reference_mode, 'cfg_reference_label': self.cfg_reference_label, 'cfg_reference_instruction': self.cfg_reference_instruction, 'cfg_source_instruction': self.cfg_source_instruction, 'lambda_plus': self.cfg_lambda_plus, 'lambda_minus': self.cfg_lambda_minus, 'cfg_event_gate_beta': self.cfg_event_gate_beta, 'cfg_event_gate_threshold': self.cfg_event_gate_threshold, 'cfg_event_gate_stop_threshold': self.cfg_event_gate_stop_threshold, 'cfg_conflict_cos_threshold': self.cfg_conflict_cos_threshold, 'cfg_conflict_mode': self.cfg_conflict_mode, 'cfg_gate_mode': self.cfg_gate_mode, 'cfg_gate_on_threshold': self.cfg_gate_on_threshold, 'cfg_gate_off_threshold': self.cfg_gate_off_threshold, 'cfg_gate_ema_decay': self.cfg_gate_ema_decay, 'cfg_gate_state': self.cfg_gate_state, 'cfg_gate_ema_value': self.cfg_gate_ema_value, 'cfg_source_memory_decay': self.cfg_source_memory_decay, 'cfg_source_memory_mode': self.cfg_source_memory_mode, 'cfg_soft_mask_temperature': self.cfg_soft_mask_temperature, 'cfg_soft_mask_dominance_eta': self.cfg_soft_mask_dominance_eta, 'cfg_norm_balance_clip_min': self.cfg_norm_balance_clip_min, 'cfg_norm_balance_clip_max': self.cfg_norm_balance_clip_max, 'cfg_novelty_boost_scale': self.cfg_novelty_boost_scale, 'cfg_hybrid_reference_decay': self.cfg_hybrid_reference_decay, 'cfg_trust_region_enabled': self.cfg_trust_region_enabled, 'cfg_trust_region_coeff': self.cfg_trust_region_coeff, 'continuity_guard': self.cfg_continuity_guard, 'switch_window': self.cfg_switch_window, 'target_branch': self.cfg_target_branch, 'null_branch': self.cfg_null_branch, 'source_branch': self.cfg_source_branch, 'cfg_activation_window_mode': self.cfg_activation_window_mode, 'cfg_projection_mode': self.cfg_projection_mode, 'cfg_projection_strength': self.cfg_projection_strength, 'cfg_projection_prefix_steps': self.cfg_projection_prefix_steps, 'uncond_instruction': self.uncond_instruction, 'uncond_visual_mode': self.uncond_visual_mode, 'uncond_blur_kernel_size': self.uncond_blur_kernel_size, 'uncond_blur_sigma': self.uncond_blur_sigma, 'uncond_blur_active': uncond_blur_active, 'uncond_first_frame_mode': 'independent_blurred' if uncond_blur_active else 'shared_clear', 'switch_trace_enabled': self.switch_trace_enabled, 'trace_stats_requested': list(self.trace_stat_keys), 'cfg_whitebox_enabled': self.cfg_whitebox_enabled, 'cfg_whitebox_save_vectors': self.cfg_whitebox_save_vectors, 'cfg_whitebox_first_k': self.cfg_whitebox_first_k, 'iti2_bridge_requested': bridge_requested, 'iti2_bridge_enabled': self.bridge_search_enabled, 'iti2_bridge_window_inferences': self.bridge_inferences_remaining, 'iti2_candidate_scales': list(self.bridge_candidate_scales), 'iti2_blend_weights': list(self.bridge_blend_weights), 'iti2_bridge_prefix_steps': self.bridge_prefix_steps, 'iti2_projection_strength': self.bridge_projection_strength, 'iti2_target_weight': self.bridge_target_weight, 'iti2_source_guard_weight': self.bridge_source_guard_weight, 'iti2_boundary_weight': self.bridge_boundary_weight, 'iti2_uncertainty_weight': self.bridge_uncertainty_weight, 'iti2_source_action_count': 0 if self.bridge_source_actions is None else int(self.bridge_source_actions.shape[0]), 'iti2_source_action_preview': self._build_bridge_source_preview()}
        if self.cfg_reference_mode in {'dual_residual_counterfactual', 'event_triggered_source_full', 'conflict_aware_event_triggered', 'conflict_aware_sparse_source_repulsion', 'alignment_gated_source_repulsion', 'semantic_metric_source_repulsion', 'dimension_masked_source_repulsion', 'source_memory_trace_cfg', 'hysteretic_event_cfg', 'soft_sign_conflict_cfg', 'norm_balanced_cfg', 'target_novelty_boost_cfg', 'hybrid_reference_takeover_cfg', 'novelty_hybrid_takeover_cfg', 'novelty_conflict_mask_cfg'} and (not self.cfg_source_instruction):
            self.switch_metadata['counterfactual_warning'] = 'source_instruction_missing_at_switch'
        if self.cfg_continuity_guard == 'on' and self.bridge_source_actions is None:
            self.switch_metadata['continuity_guard_warning'] = 'requested_without_source_action_history'
        if self.progress_trigger_metadata:
            self.switch_metadata['learned_progress_trigger'] = dict(self.progress_trigger_metadata)
        if bridge_warning is not None:
            self.switch_metadata['iti2_bridge_warning'] = bridge_warning
        return dict(self.switch_metadata)

    def record_executed_action(self, action: np.ndarray):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self.last_executed_action = action_arr.copy()
        if self.offline_trace_enabled:
            self.offline_episode_action_trace.append(action_arr.tolist())
        if not self.switch_metadata:
            return
        if not self.switch_trace_enabled and (not self.cfg_whitebox_enabled):
            return
        self.post_switch_action_trace.append(action_arr.tolist())

    def get_switch_trace_payload(self) -> Dict[str, Any]:
        progress_metadata = dict(self.progress_trace_metadata)
        if self.progress_trigger_metadata:
            progress_metadata['trigger'] = dict(self.progress_trigger_metadata)
        if self.progress_dry_run_trigger_metadata:
            progress_metadata['would_trigger'] = dict(self.progress_dry_run_trigger_metadata)
        offline_payload = self._build_offline_trace_payload()
        if not self.switch_metadata and (not progress_metadata) and (offline_payload is None):
            return {}
        metadata = dict(self.switch_metadata)
        if progress_metadata:
            metadata['learned_progress'] = progress_metadata
        payload = {}
        if metadata:
            payload['switch_metadata'] = metadata
        if self.progress_trace_history:
            payload['progress_trace'] = list(self.progress_trace_history)
        if self.switch_metadata:
            trace_stats = self._compute_trace_stats()
            metadata['post_switch_action_count'] = len(self.post_switch_action_trace)
            metadata['cfg_scale_history'] = list(self.switch_cfg_scale_history)
            metadata['cfg_gate_history'] = list(self.cfg_gate_history)
            metadata['cfg_gate_ema_value'] = float(self.cfg_gate_ema_value)
            metadata['cfg_gate_state'] = bool(self.cfg_gate_state)
            metadata['all_finite'] = bool(trace_stats.get('all_finite', True))
            metadata['has_nan'] = bool(trace_stats.get('has_nan', False))
            metadata['has_inf'] = bool(trace_stats.get('has_inf', False))
            metadata['numerical_stability_violation_count'] = int(trace_stats.get('numerical_stability_violation_count', 0))
            if self.cfg_whitebox_enabled and self.cfg_whitebox_payload:
                cfg_whitebox_payload = dict(self.cfg_whitebox_payload)
                cfg_whitebox_payload['post_switch_action_preview'] = self._build_post_switch_action_preview()
                if cfg_whitebox_payload.get('cf_branch_payloads'):
                    metadata['cf_branch_payloads'] = dict(cfg_whitebox_payload['cf_branch_payloads'])
                metadata['cfg_whitebox'] = cfg_whitebox_payload
            if self.bridge_selection_history:
                metadata['iti2_bridge'] = {'window_count': len(self.bridge_selection_history), 'windows': list(self.bridge_selection_history)}
            payload['trace_stats'] = trace_stats
            if self.switch_trace_enabled:
                payload['action_trace'] = list(self.post_switch_action_trace)
        if offline_payload is not None:
            payload['offline_corpus'] = offline_payload['metadata']
            payload['trace_stats'] = offline_payload['trace_stats']
            payload['action_trace'] = offline_payload['action_trace']
        return payload

    def _reset_progress_state(self) -> None:
        self.progress_hidden_state = None
        self.progress_scores = []
        self.latest_progress_score = None
        self.progress_switch_consumed = False
        self.progress_trigger_metadata = {}
        self.progress_dry_run_trigger_metadata = {}
        self.progress_trace_metadata = {}
        self.progress_trace_history = []

    def _load_progress_head(self, checkpoint_path: str) -> None:
        progress_head_path = Path(__file__).resolve().parents[3] / 'Motus' / 'tools' / 'progress' / 'progress_head.py'
        if not progress_head_path.is_file():
            raise FileNotFoundError(f'Progress head loader not found: {progress_head_path}')
        spec = importlib.util.spec_from_file_location('motus_progress_head_loader', progress_head_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Failed to load progress head module spec from: {progress_head_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        load_progress_checkpoint = module.load_progress_checkpoint
        self.progress_head, metadata = load_progress_checkpoint(checkpoint_path, self.device)
        self.progress_feature_source = str(metadata.get('feature_source', 'motus_visual'))
        self.progress_enabled = True
        self._reset_progress_state()
        logger.info('Loaded progress head from %s | feature_source=%s | input_dim=%s', checkpoint_path, self.progress_feature_source, metadata.get('input_dim'))

    def _build_progress_input(self, vlm_inputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.progress_enabled or self.progress_head is None:
            return None
        if self.progress_feature_source.startswith('motus_action_decoder'):
            if self.latest_action_progress_feature is None:
                return None
            progress_input = self.latest_action_progress_feature
            if self.progress_feature_source.endswith('_plus_qpos'):
                if self.current_state_norm is None:
                    return None
                progress_input = torch.cat([progress_input, self.current_state_norm], dim=-1)
            return progress_input
        parts: List[torch.Tensor] = []
        if self.progress_feature_source != 'qpos_only':
            und_tokens = self.model.und_module.extract_und_features(vlm_inputs)
            parts.append(und_tokens.mean(dim=1))
        if self.progress_feature_source in {'motus_visual_plus_qpos', 'qpos_only'}:
            if self.current_state_norm is None:
                return None
            parts.append(self.current_state_norm)
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _update_progress_score(self, vlm_inputs: Dict[str, torch.Tensor]) -> None:
        progress_input = self._build_progress_input(vlm_inputs)
        if progress_input is None:
            return
        score_tensor, self.progress_hidden_state = self.progress_head.forward_streaming(progress_input.unsqueeze(1), self.progress_hidden_state)
        self.latest_progress_score = float(score_tensor.squeeze(0).item())
        self.progress_scores.append(self.latest_progress_score)
        self.progress_trace_metadata = {'feature_source': self.progress_feature_source, 'steps_seen': len(self.progress_scores), 'latest_progress_score': self.latest_progress_score, 'scores_tail': list(self.progress_scores[-10:])}

    def maybe_instruction_switch(self, threshold: float=0.8, consecutive_frames: int=3, min_dwell_steps: int=5, score_log_interval: int=10, switch_mode: str='live') -> Dict[str, Any]:
        normalized_switch_mode = str(switch_mode or 'live').lower()
        if normalized_switch_mode in {'learned_progress', 'enabled'}:
            normalized_switch_mode = 'live'
        if self.progress_enabled and self.latest_progress_score is not None:
            interval = max(int(score_log_interval), 1)
            if len(self.progress_scores) % interval == 0:
                logger.info('[Progress] steps=%s latest=%.4f threshold=%.4f', len(self.progress_scores), self.latest_progress_score, threshold)
        metadata = {'progress_enabled': bool(self.progress_enabled), 'switch_mode': normalized_switch_mode, 'latest_progress_score': self.latest_progress_score, 'progress_threshold': float(threshold), 'progress_consecutive_frames': int(consecutive_frames), 'progress_min_dwell': int(min_dwell_steps), 'progress_steps_seen': len(self.progress_scores), 'scores_tail': list(self.progress_scores[-10:]), 'should_switch': False, 'would_switch': False}
        if not self.progress_enabled or self.latest_progress_score is None or self.progress_switch_consumed:
            self.progress_trace_metadata = dict(metadata)
            return metadata
        consecutive_above = 0
        for value in reversed(self.progress_scores):
            if value >= threshold:
                consecutive_above += 1
            else:
                break
        criteria_met = len(self.progress_scores) >= max(int(min_dwell_steps), 1) and consecutive_above >= max(int(consecutive_frames), 1)
        metadata['consecutive_above_threshold'] = int(consecutive_above)
        metadata['would_switch'] = bool(criteria_met)
        metadata['should_switch'] = bool(criteria_met and normalized_switch_mode == 'live')
        if criteria_met:
            metadata['trigger_frame'] = int(len(self.progress_scores) - 1)
            metadata['switch_source'] = 'learned_progress'
            if normalized_switch_mode == 'live':
                self.progress_switch_consumed = True
                self.progress_trigger_metadata = dict(metadata)
            elif not self.progress_dry_run_trigger_metadata:
                self.progress_dry_run_trigger_metadata = dict(metadata)
        self.progress_trace_metadata = dict(metadata)
        self.progress_trace_history.append({'step': int(max(len(self.progress_scores) - 1, 0)), 'score': float(self.latest_progress_score), 'threshold': float(threshold), 'consecutive_above_threshold': int(consecutive_above), 'would_switch': bool(metadata['would_switch']), 'should_switch': bool(metadata['should_switch']), 'switch_mode': str(metadata['switch_mode'])})
        return metadata

    def probe_instruction_switch_router(self, target_instruction: str, decision_mode: str='online_probe_v1', guidance_scale: float=1.0, low_consistency_threshold: float=0.55, mean_consistency_threshold: float=0.9, high_delta_l2_threshold: float=3.0, min_delta_l2_for_switch: float=1.25, max_boundary_cost_delta: float=0.75, boundary_cost_mean_consistency_guard: float=0.96, target_source_cosine_mean_mean_threshold: Optional[float]=None, target_source_alignment_min_delta_l2: Optional[float]=None, legacy_rules_enabled: bool=True, probe_cfg_reference_mode: str='runtime_uncond', probe_lambda_plus: float=1.0, probe_lambda_minus: float=0.0, default_decision: str='raw', source: str='', threshold_selection: str='', metric: str='', score_threshold: Optional[float]=None, semantic_dominance_stats: Optional[Dict[str, Any]]=None, semantic_retention_bonus_weight: float=0.0, target_null_delta_l2_weight: float=0.0, semantic_positive_retention_penalty_weight: float=0.0, target_novelty_l1_weight: float=0.0, semantic_v4_score_weight: float=1.0, semantic_source_distinctiveness_weight: float=0.0, motor_compatibility_weight: float=0.0, recoverability_penalty_weight: float=0.0, semantic_positive_v3_penalty_weight: float=0.0) -> Dict[str, Any]:
        router_mode = str(decision_mode or 'online_probe_v1').strip().lower()
        semantic_stats = dict(semantic_dominance_stats or {})
        metadata = {'adaptive_switch_router_enabled': True, 'adaptive_switch_router_decision_mode': router_mode, 'adaptive_switch_router_default_decision': str(default_decision or 'raw'), 'adaptive_switch_router_source': str(source or ''), 'adaptive_switch_router_metric': str(metric or ''), 'adaptive_switch_router_threshold_selection': str(threshold_selection or ''), 'adaptive_switch_router_target_instruction': str(target_instruction or ''), 'adaptive_switch_router_decision': str(default_decision or 'raw'), 'adaptive_switch_router_decision_reason': 'default', 'adaptive_switch_router_probe_available': False, 'adaptive_switch_router_thresholds': {'low_consistency_threshold': float(low_consistency_threshold), 'mean_consistency_threshold': float(mean_consistency_threshold), 'high_delta_l2_threshold': float(high_delta_l2_threshold), 'min_delta_l2_for_switch': float(min_delta_l2_for_switch), 'max_boundary_cost_delta': float(max_boundary_cost_delta), 'boundary_cost_mean_consistency_guard': float(boundary_cost_mean_consistency_guard), 'target_source_cosine_mean_mean_threshold': float(target_source_cosine_mean_mean_threshold) if target_source_cosine_mean_mean_threshold is not None else None, 'target_source_alignment_min_delta_l2': float(target_source_alignment_min_delta_l2) if target_source_alignment_min_delta_l2 is not None else None, 'legacy_rules_enabled': bool(legacy_rules_enabled), 'probe_cfg_reference_mode': str(probe_cfg_reference_mode or 'runtime_uncond'), 'probe_lambda_plus': float(probe_lambda_plus), 'probe_lambda_minus': float(probe_lambda_minus), 'score_threshold': float(score_threshold) if score_threshold is not None else None, 'semantic_dominance_stats': semantic_stats or None, 'semantic_retention_bonus_weight': float(semantic_retention_bonus_weight), 'target_null_delta_l2_weight': float(target_null_delta_l2_weight), 'semantic_positive_retention_penalty_weight': float(semantic_positive_retention_penalty_weight), 'target_novelty_l1_weight': float(target_novelty_l1_weight), 'semantic_v4_score_weight': float(semantic_v4_score_weight), 'semantic_source_distinctiveness_weight': float(semantic_source_distinctiveness_weight), 'motor_compatibility_weight': float(motor_compatibility_weight), 'recoverability_penalty_weight': float(recoverability_penalty_weight), 'semantic_positive_v3_penalty_weight': float(semantic_positive_v3_penalty_weight)}}
        if router_mode not in {'online_probe_v1', 'semantic_dominance_v1', 'semantic_dominance_v3', 'semantic_dominance_v4_min', 'semantic_dominance_v5_safe_takeover', 'semantic_dominance_v6_recoverability', 'semantic_dominance_v6_saturation_guard'}:
            metadata['adaptive_switch_router_probe_warning'] = f'unsupported_decision_mode:{router_mode}'
            return metadata
        if len(self.obs_cache) == 0 or self.current_state is None:
            metadata['adaptive_switch_router_probe_warning'] = 'missing_observation_or_state'
            return metadata
        num_inference_steps = int(self.config_dict['model']['inference']['num_inference_timesteps'])
        current_frame = self.obs_cache[-1]
        scene_prefix = self._build_scene_prefix()
        target_instruction_full = f'{scene_prefix}{target_instruction}'
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        target_t5_out = self.t5_encoder([target_instruction_full], self.device)
        target_t5_list = self._ensure_t5_list(target_t5_out)
        target_vlm_inputs = self._preprocess_vlm_messages(target_instruction_full, first_frame_pil)
        uncond_instruction = scene_prefix
        uncond_t5_out = self.t5_encoder([uncond_instruction], self.device)
        uncond_t5_list = self._ensure_t5_list(uncond_t5_out)
        uncond_vlm_inputs = self._preprocess_vlm_messages(uncond_instruction, first_frame_pil)
        source_alignment_probe_enabled = target_source_cosine_mean_mean_threshold is not None or router_mode in {'semantic_dominance_v1', 'semantic_dominance_v3', 'semantic_dominance_v4_min', 'semantic_dominance_v5_safe_takeover', 'semantic_dominance_v6_recoverability', 'semantic_dominance_v6_saturation_guard'}
        source_t5_list = None
        source_vlm_inputs = None
        if source_alignment_probe_enabled:
            source_instruction_full = f"{scene_prefix}{str(self.current_instruction or '')}"
            if not str(self.current_instruction or ''):
                metadata['adaptive_switch_router_probe_warning'] = 'missing_source_instruction'
                return metadata
            source_t5_out = self.t5_encoder([source_instruction_full], self.device)
            source_t5_list = self._ensure_t5_list(source_t5_out)
            source_vlm_inputs = self._preprocess_vlm_messages(source_instruction_full, first_frame_pil)
        raw_actions_tensor = None
        with torch.inference_mode():
            if legacy_rules_enabled:
                _, raw_actions_tensor = self.model.inference_step(first_frame=current_frame, state=self.current_state, num_inference_steps=num_inference_steps, language_embeddings=target_t5_list, vlm_inputs=[target_vlm_inputs], prior_action=None, forcing_steps=0)
            _, cfg_actions_tensor, cfg_debug_payload = self.model.inference_step_cfg(first_frame=current_frame, state=self.current_state, num_inference_steps=num_inference_steps, language_embeddings=target_t5_list, vlm_inputs=[target_vlm_inputs], uncond_language_embeddings=uncond_t5_list, uncond_vlm_inputs=[uncond_vlm_inputs], uncond_first_frame=None, source_language_embeddings=source_t5_list, source_vlm_inputs=[source_vlm_inputs] if source_vlm_inputs is not None else None, cfg_reference_mode=str(probe_cfg_reference_mode or 'runtime_uncond') if source_alignment_probe_enabled else 'runtime_uncond', lambda_plus=float(probe_lambda_plus if source_alignment_probe_enabled else 1.0), lambda_minus=float(probe_lambda_minus if source_alignment_probe_enabled else 0.0), guidance_scale=float(guidance_scale), prior_action=None, forcing_steps=0, return_cfg_debug=True, save_cfg_vectors=False)
        cfg_actions = self._cfg_tensor_to_action_array(cfg_actions_tensor)
        action_target = self._cfg_tensor_to_action_array(cfg_debug_payload.get('predicted_actions_target', cfg_debug_payload.get('predicted_actions_cond')))
        action_null = self._cfg_tensor_to_action_array(cfg_debug_payload.get('predicted_actions_null', cfg_debug_payload.get('predicted_actions_uncond')))
        action_source = self._cfg_tensor_to_action_array(cfg_debug_payload.get('predicted_actions_source'))
        raw_actions = self._cfg_tensor_to_action_array(raw_actions_tensor) if raw_actions_tensor is not None else None
        if action_target is None or action_null is None or cfg_actions is None:
            metadata['adaptive_switch_router_probe_warning'] = 'missing_probe_actions'
            return metadata
        guided_action_magnitude = self._build_action_sequence_magnitude_diagnostics(cfg_actions)
        target_action_magnitude = self._build_action_sequence_magnitude_diagnostics(action_target)
        residual = action_target - action_null
        consistency = self._compute_action_sequence_cosine_stats(residual)
        target_null_delta = self._build_action_delta_diagnostics(action_target, action_null) or {}
        source_actions = self._capture_bridge_source_actions(list(self.action_cache)) if legacy_rules_enabled else None
        raw_boundary_cost = self._compute_boundary_cost_against_source(raw_actions, source_actions) if legacy_rules_enabled and raw_actions is not None else None
        cfg_boundary_cost = self._compute_boundary_cost_against_source(cfg_actions, source_actions) if legacy_rules_enabled else None
        guided_source_action_cosine = self._build_action_pair_cosine_diagnostics(cfg_actions, action_source)
        target_source_action_cosine = self._build_action_pair_cosine_diagnostics(action_target, action_source)
        guided_target_action_cosine = self._build_action_pair_cosine_diagnostics(cfg_actions, action_target)
        boundary_cost_delta = float(cfg_boundary_cost - raw_boundary_cost) if raw_boundary_cost is not None and cfg_boundary_cost is not None else None
        delta_l2 = float(target_null_delta.get('delta_l2', 0.0) or 0.0)
        resolved_default_decision = str(default_decision or 'raw').strip().lower()
        if router_mode in {'semantic_dominance_v1', 'semantic_dominance_v3', 'semantic_dominance_v4_min', 'semantic_dominance_v5_safe_takeover', 'semantic_dominance_v6_recoverability', 'semantic_dominance_v6_saturation_guard'}:
            decision = 'switch_aware' if resolved_default_decision == 'switch_aware' else 'raw'
            decision_reason = 'default'
        else:
            decision = 'switch_aware'
            decision_reason = 'consistent_residual_probe'
        target_source_cosine_mean_mean = None
        target_source_cosine_mean_min = None
        guided_source_cosine_mean_mean = None
        guided_target_cosine_mean_mean = None
        takeover_gap_mean = None
        semantic_dominance_z_guided_target_action = None
        semantic_dominance_z_guided_target_cosine = None
        semantic_dominance_z_source_retention = None
        semantic_dominance_score = None
        semantic_dominance_retention_bonus = None
        semantic_dominance_v2_score = None
        semantic_dominance_target_null_delta_l2_penalty = None
        semantic_dominance_v3_score = None
        semantic_dominance_positive_retention_penalty = None
        semantic_dominance_target_novelty_l1_penalty = None
        semantic_dominance_v4_score = None
        semantic_dominance_z_target_source_cosine = None
        semantic_dominance_z_target_source_action = None
        semantic_dominance_z_guided_action_l2_delta = None
        semantic_dominance_v5_score = None
        semantic_dominance_recoverability_penalty = None
        semantic_dominance_v6_score = None
        semantic_dominance_positive_v3_penalty = None
        semantic_dominance_v6_saturation_guard_score = None
        if source_alignment_probe_enabled:
            raw_step_payloads = cfg_debug_payload.get('denoise_step_payloads') or []
            wrapped_inference_payloads = cfg_debug_payload.get('cfg_inference_diagnostics') or []
            target_source_cosines = []
            guided_source_cosines = []
            guided_target_cosines = []
            takeover_gaps = []

            def _accumulate_source_alignment(step_payload: dict[str, Any]) -> None:
                if 'target_source_cosine_mean' in step_payload:
                    target_source_cosines.append(float(step_payload['target_source_cosine_mean']))
                if 'guided_source_cosine_mean' in step_payload:
                    guided_source_cosines.append(float(step_payload['guided_source_cosine_mean']))
                if 'guided_target_cosine_mean' in step_payload:
                    guided_target_cosines.append(float(step_payload['guided_target_cosine_mean']))
                if 'takeover_gap' in step_payload:
                    takeover_gaps.append(float(step_payload['takeover_gap']))
            for step_payload in raw_step_payloads:
                if isinstance(step_payload, dict):
                    _accumulate_source_alignment(step_payload)
            for inference_payload in wrapped_inference_payloads:
                for step_payload in inference_payload.get('denoise_step_payloads') or []:
                    if isinstance(step_payload, dict):
                        _accumulate_source_alignment(step_payload)
            if not target_source_cosines:
                metadata['adaptive_switch_router_probe_warning'] = 'missing_source_alignment_probe'
                return metadata
            target_source_cosine_mean_mean = float(sum(target_source_cosines) / len(target_source_cosines))
            target_source_cosine_mean_min = float(min(target_source_cosines))
            if guided_source_cosines:
                guided_source_cosine_mean_mean = float(sum(guided_source_cosines) / len(guided_source_cosines))
            if guided_target_cosines:
                guided_target_cosine_mean_mean = float(sum(guided_target_cosines) / len(guided_target_cosines))
            if takeover_gaps:
                takeover_gap_mean = float(sum(takeover_gaps) / len(takeover_gaps))
            if router_mode == 'online_probe_v1' and target_source_cosine_mean_mean_threshold is not None and (target_source_cosine_mean_mean >= float(target_source_cosine_mean_mean_threshold)) and (delta_l2 >= float(target_source_alignment_min_delta_l2 or 0.0)):
                decision = 'raw'
                decision_reason = 'target_source_alignment_high'
            elif router_mode == 'online_probe_v1' and (not legacy_rules_enabled):
                decision = 'switch_aware'
                decision_reason = 'target_source_alignment_guard_pass'
        if router_mode in {'semantic_dominance_v1', 'semantic_dominance_v3', 'semantic_dominance_v4_min', 'semantic_dominance_v5_safe_takeover', 'semantic_dominance_v6_recoverability', 'semantic_dominance_v6_saturation_guard'}:
            guided_target_action_cosine_mean = float(guided_target_action_cosine.get('mean')) if guided_target_action_cosine is not None and guided_target_action_cosine.get('mean') is not None else None
            source_retention_action_cosine_mean = float(guided_source_action_cosine.get('mean')) if guided_source_action_cosine is not None and guided_source_action_cosine.get('mean') is not None else None

            def _semantic_stat(name: str) -> tuple[Optional[float], Optional[float]]:
                payload = semantic_stats.get(name)
                if not isinstance(payload, dict):
                    return (None, None)
                mu = payload.get('mu')
                sigma = payload.get('sigma')
                return (float(mu) if mu is not None else None, float(sigma) if sigma is not None else None)

            def _safe_z(value: Optional[float], name: str) -> Optional[float]:
                mu, sigma = _semantic_stat(name)
                if value is None or mu is None or sigma is None or (sigma == 0.0):
                    return None
                return (float(value) - mu) / sigma
            semantic_dominance_z_guided_target_action = _safe_z(guided_target_action_cosine_mean, 'guided_target_action_cosine_mean')
            semantic_dominance_z_guided_target_cosine = _safe_z(guided_target_cosine_mean_mean, 'guided_target_cosine_mean_mean')
            semantic_dominance_z_source_retention = _safe_z(source_retention_action_cosine_mean, 'R_e_step')
            if semantic_dominance_z_guided_target_action is None or semantic_dominance_z_guided_target_cosine is None or semantic_dominance_z_source_retention is None:
                metadata['adaptive_switch_router_probe_warning'] = 'missing_semantic_dominance_features_or_stats'
                decision = 'switch_aware' if resolved_default_decision == 'switch_aware' else 'raw'
                decision_reason = 'semantic_dominance_missing_features_or_stats'
            else:
                semantic_dominance_score = semantic_dominance_z_guided_target_action + semantic_dominance_z_guided_target_cosine - semantic_dominance_z_source_retention
                semantic_dominance_retention_bonus = float(semantic_retention_bonus_weight) * max(0.0, -semantic_dominance_z_source_retention)
                semantic_dominance_v2_score = semantic_dominance_z_guided_target_action + semantic_dominance_z_guided_target_cosine + semantic_dominance_retention_bonus
                semantic_dominance_target_null_delta_l2_penalty = float(target_null_delta_l2_weight) * float(delta_l2)
                semantic_dominance_v3_score = semantic_dominance_v2_score - semantic_dominance_target_null_delta_l2_penalty
                target_novelty_l1_mean = float(target_null_delta.get('delta_l1_mean', 0.0) or 0.0)
                semantic_dominance_positive_retention_penalty = float(semantic_positive_retention_penalty_weight) * max(0.0, semantic_dominance_z_source_retention)
                semantic_dominance_target_novelty_l1_penalty = float(target_novelty_l1_weight) * target_novelty_l1_mean
                semantic_dominance_v4_score = semantic_dominance_v3_score - semantic_dominance_positive_retention_penalty - semantic_dominance_target_novelty_l1_penalty
                semantic_dominance_z_target_source_cosine = _safe_z(target_source_cosine_mean_mean, 'target_source_cosine_mean_mean')
                semantic_dominance_z_target_source_action = _safe_z(float(target_source_action_cosine.get('mean')) if target_source_action_cosine is not None and target_source_action_cosine.get('mean') is not None else None, 'target_source_action_cosine_mean')
                semantic_dominance_z_guided_action_l2_delta = _safe_z(guided_action_magnitude.get('l2_delta') if guided_action_magnitude else None, 'guided_action_l2_delta')
                if semantic_dominance_v4_score is not None and semantic_dominance_z_target_source_cosine is not None and (semantic_dominance_z_target_source_action is not None):
                    semantic_dominance_v5_score = float(semantic_v4_score_weight) * semantic_dominance_v4_score + float(semantic_source_distinctiveness_weight) * max(0.0, -semantic_dominance_z_target_source_cosine) + float(motor_compatibility_weight) * semantic_dominance_z_target_source_action
                if semantic_dominance_v5_score is not None and semantic_dominance_z_guided_action_l2_delta is not None:
                    semantic_dominance_recoverability_penalty = float(recoverability_penalty_weight) * semantic_dominance_z_guided_action_l2_delta
                    semantic_dominance_v6_score = semantic_dominance_v5_score - semantic_dominance_recoverability_penalty
                if semantic_dominance_v5_score is not None:
                    semantic_dominance_positive_v3_penalty = float(semantic_positive_v3_penalty_weight) * max(0.0, float(semantic_dominance_v3_score or 0.0))
                    semantic_dominance_v6_saturation_guard_score = semantic_dominance_v5_score - semantic_dominance_positive_v3_penalty
                resolved_score_threshold = float(score_threshold if score_threshold is not None else 0.0)
                resolved_semantic_score = semantic_dominance_v6_saturation_guard_score if router_mode == 'semantic_dominance_v6_saturation_guard' else semantic_dominance_v6_score if router_mode == 'semantic_dominance_v6_recoverability' else semantic_dominance_v5_score if router_mode == 'semantic_dominance_v5_safe_takeover' else semantic_dominance_v4_score if router_mode == 'semantic_dominance_v4_min' else semantic_dominance_v3_score if router_mode == 'semantic_dominance_v3' else semantic_dominance_score
                if resolved_semantic_score is None:
                    metadata['adaptive_switch_router_probe_warning'] = 'missing_semantic_dominance_v5_or_v6_features_or_stats'
                    decision = 'switch_aware' if resolved_default_decision == 'switch_aware' else 'raw'
                    decision_reason = 'semantic_dominance_v5_or_v6_missing_features_or_stats'
                    resolved_semantic_score = float('-inf')
                decision = 'switch_aware' if resolved_semantic_score > resolved_score_threshold else 'raw'
                decision_reason = ('semantic_dominance_v6_saturation_guard_score_above_threshold' if router_mode == 'semantic_dominance_v6_saturation_guard' else 'semantic_dominance_v6_score_above_threshold' if router_mode == 'semantic_dominance_v6_recoverability' else 'semantic_dominance_v5_score_above_threshold' if router_mode == 'semantic_dominance_v5_safe_takeover' else 'semantic_dominance_v4_score_above_threshold' if router_mode == 'semantic_dominance_v4_min' else 'semantic_dominance_v3_score_above_threshold' if router_mode == 'semantic_dominance_v3' else 'semantic_dominance_score_above_threshold') if decision == 'switch_aware' else 'semantic_dominance_v6_saturation_guard_score_below_threshold' if router_mode == 'semantic_dominance_v6_saturation_guard' else 'semantic_dominance_v6_score_below_threshold' if router_mode == 'semantic_dominance_v6_recoverability' else 'semantic_dominance_v5_score_below_threshold' if router_mode == 'semantic_dominance_v5_safe_takeover' else 'semantic_dominance_v4_score_below_threshold' if router_mode == 'semantic_dominance_v4_min' else 'semantic_dominance_v3_score_below_threshold' if router_mode == 'semantic_dominance_v3' else 'semantic_dominance_score_below_threshold'
        if router_mode == 'online_probe_v1' and legacy_rules_enabled and (decision == 'switch_aware'):
            if delta_l2 < float(min_delta_l2_for_switch):
                decision = 'raw'
                decision_reason = 'weak_residual_signal'
            elif float(consistency['min']) < float(low_consistency_threshold):
                decision = 'raw'
                decision_reason = 'low_residual_direction_consistency_min'
            elif delta_l2 >= float(high_delta_l2_threshold) and float(consistency['mean']) < float(mean_consistency_threshold):
                decision = 'raw'
                decision_reason = 'high_residual_magnitude_but_low_mean_consistency'
            elif boundary_cost_delta is not None and boundary_cost_delta > float(max_boundary_cost_delta) and (float(consistency['mean']) < float(boundary_cost_mean_consistency_guard)):
                decision = 'raw'
                decision_reason = 'cfg_boundary_cost_too_high'
        metadata.update({'adaptive_switch_router_probe_available': True, 'adaptive_switch_router_decision': decision, 'adaptive_switch_router_decision_reason': decision_reason, 'adaptive_switch_router_base_mode': router_mode, 'adaptive_switch_router_resolved_mode': 'none' if decision == 'raw' else 'legacy', 'adaptive_switch_router_probe_features': {'target_null_delta_l2': delta_l2, 'target_null_delta_l1_mean': float(target_null_delta.get('delta_l1_mean', 0.0) or 0.0), 'residual_step_cosine_mean': float(consistency['mean']), 'residual_step_cosine_min': float(consistency['min']), 'residual_step_cosine_std': float(consistency['std']), 'raw_boundary_cost': float(raw_boundary_cost) if raw_boundary_cost is not None else None, 'cfg_boundary_cost': float(cfg_boundary_cost) if cfg_boundary_cost is not None else None, 'cfg_minus_raw_boundary_cost': boundary_cost_delta, 'target_source_cosine_mean_mean': target_source_cosine_mean_mean, 'target_source_cosine_mean_min': target_source_cosine_mean_min, 'guided_source_cosine_mean_mean': guided_source_cosine_mean_mean, 'guided_target_cosine_mean_mean': guided_target_cosine_mean_mean, 'takeover_gap_mean': takeover_gap_mean, 'source_action_count': 0 if source_actions is None else int(source_actions.shape[0]), 'source_retention_action_cosine_mean': float(guided_source_action_cosine.get('mean')) if guided_source_action_cosine is not None else None, 'source_retention_action_cosine_min': float(guided_source_action_cosine.get('min')) if guided_source_action_cosine is not None else None, 'target_source_action_cosine_mean': float(target_source_action_cosine.get('mean')) if target_source_action_cosine is not None else None, 'target_source_action_cosine_min': float(target_source_action_cosine.get('min')) if target_source_action_cosine is not None else None, 'guided_target_action_cosine_mean': float(guided_target_action_cosine.get('mean')) if guided_target_action_cosine is not None else None, 'guided_target_action_cosine_min': float(guided_target_action_cosine.get('min')) if guided_target_action_cosine is not None else None, 'semantic_dominance_z_guided_target_action': semantic_dominance_z_guided_target_action, 'semantic_dominance_z_guided_target_cosine': semantic_dominance_z_guided_target_cosine, 'semantic_dominance_z_source_retention': semantic_dominance_z_source_retention, 'semantic_dominance_score': semantic_dominance_score, 'semantic_dominance_retention_bonus': semantic_dominance_retention_bonus, 'semantic_dominance_v2_score': semantic_dominance_v2_score, 'semantic_dominance_target_null_delta_l2_penalty': semantic_dominance_target_null_delta_l2_penalty, 'semantic_dominance_v3_score': semantic_dominance_v3_score, 'semantic_dominance_positive_retention_penalty': semantic_dominance_positive_retention_penalty, 'semantic_dominance_target_novelty_l1_penalty': semantic_dominance_target_novelty_l1_penalty, 'semantic_dominance_v4_score': semantic_dominance_v4_score, 'semantic_dominance_z_target_source_cosine': semantic_dominance_z_target_source_cosine, 'semantic_dominance_z_target_source_action': semantic_dominance_z_target_source_action, 'semantic_dominance_z_guided_action_l2_delta': semantic_dominance_z_guided_action_l2_delta, 'semantic_dominance_v5_score': semantic_dominance_v5_score, 'semantic_dominance_recoverability_penalty': semantic_dominance_recoverability_penalty, 'semantic_dominance_v6_score': semantic_dominance_v6_score, 'semantic_dominance_positive_v3_penalty': semantic_dominance_positive_v3_penalty, 'semantic_dominance_v6_saturation_guard_score': semantic_dominance_v6_saturation_guard_score, 'semantic_dominance_threshold': float(score_threshold) if score_threshold is not None else None, 'target_novelty_l1_mean': float(target_null_delta.get('delta_l1_mean', 0.0) or 0.0), 'guided_action_l2_first': guided_action_magnitude.get('l2_first') if guided_action_magnitude else None, 'guided_action_l2_last': guided_action_magnitude.get('l2_last') if guided_action_magnitude else None, 'guided_action_l2_delta': guided_action_magnitude.get('l2_delta') if guided_action_magnitude else None, 'guided_action_l2_slope': guided_action_magnitude.get('l2_slope') if guided_action_magnitude else None, 'target_action_l2_first': target_action_magnitude.get('l2_first') if target_action_magnitude else None, 'target_action_l2_last': target_action_magnitude.get('l2_last') if target_action_magnitude else None, 'target_action_l2_delta': target_action_magnitude.get('l2_delta') if target_action_magnitude else None, 'guided_action_count': 0 if cfg_actions is None else int(cfg_actions.shape[0]), 'target_action_count': 0 if action_target is None else int(action_target.shape[0]), 'source_probe_action_count': 0 if action_source is None else int(action_source.shape[0])}})
        return metadata

    def _load_model(self) -> Motus:
        logger.info(f'Initializing Motus model from config (no pretrained backbones)')
        config = self._create_model_config()
        model = Motus(config)
        model = model.to(self.device)
        try:
            logger.info(f'Loading checkpoint from {self.checkpoint_path}')
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info('Model checkpoint loaded successfully')
        except Exception as e:
            logger.error(f'Failed to load checkpoint: {e}')
            raise
        model.eval()
        return model

    def _build_scene_prefix(self) -> str:
        return 'The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: '

    def _ensure_t5_list(self, t5_out):
        if isinstance(t5_out, torch.Tensor):
            return [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        if isinstance(t5_out, list):
            return t5_out
        raise ValueError('Unexpected T5 encoder output format')

    def _create_model_config(self) -> MotusConfig:
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']
        vae_path = os.path.join(self.wan_path, 'Wan2.2_VAE.pth')
        vlm_checkpoint_path = self.vlm_path
        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']
        config = MotusConfig(wan_checkpoint_path=self.wan_path, vae_path=vae_path, wan_config_path=self.wan_path, video_precision='bfloat16', vlm_checkpoint_path=vlm_checkpoint_path, und_expert_hidden_size=512, und_expert_ffn_dim_multiplier=4, und_expert_norm_eps=1e-05, und_layers_to_extract=None, vlm_adapter_input_dim=2048, vlm_adapter_projector_type='mlp3x_silu', num_layers=30, action_state_dim=common['state_dim'], action_dim=common['action_dim'], action_expert_dim=hidden_size, action_expert_ffn_dim_multiplier=ffn_multiplier, action_expert_norm_eps=1e-06, global_downsample_rate=common['global_downsample_rate'], video_action_freq_ratio=common['video_action_freq_ratio'], num_video_frames=common['num_video_frames'], video_loss_weight=1.0, action_loss_weight=1.0, batch_size=1, video_height=common['video_height'], video_width=common['video_width'], load_pretrained_backbones=False, training_mode='finetune', device=self.device)
        return config

    def update_obs(self, observation: Dict[str, Any]):
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and ('right_camera' in obs_data):
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError('Missing camera data')
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError('No visual observation found')
        target_size = (self.config_dict['common']['video_height'], self.config_dict['common']['video_width'])
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image
        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        self.obs_cache.append(image_tensor.to(self.device))
        state = observation['joint_action']['vector']
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()
        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)

    def get_action(self, instruction: str=None) -> List[np.ndarray]:
        start_time = time.time()
        if len(self.obs_cache) == 0:
            raise ValueError('No observations in cache. Call update_obs first.')
        if self.current_state is None:
            raise ValueError('No robot state available. Call update_obs first.')
        current_frame = self.obs_cache[-1]
        t5_start = time.time()
        scene_prefix = 'The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: '
        instruction = f'{scene_prefix}{self.current_instruction}'
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError('Unexpected T5 encoder output format')
        t5_time = time.time() - t5_start
        vlm_start = time.time()
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)
        uncond_first_frame_pil = first_frame_pil
        uncond_first_frame = None
        if self.cfg_enabled and self.uncond_visual_mode == 'blurred' and (self.uncond_blur_kernel_size > 1):
            uncond_first_frame_pil = self._apply_gaussian_blur(first_frame_pil, kernel_size=self.uncond_blur_kernel_size, sigma=self.uncond_blur_sigma)
            uncond_first_frame = self._pil_to_tensor(uncond_first_frame_pil).unsqueeze(0).to(current_frame.device)
            if self.switch_metadata:
                self.switch_metadata['uncond_visual_mode_applied'] = 'blurred'
                self.switch_metadata['uncond_first_frame_mode'] = 'independent_blurred'
        elif self.switch_metadata:
            self.switch_metadata['uncond_visual_mode_applied'] = 'clear'
            self.switch_metadata['uncond_first_frame_mode'] = 'shared_clear'
        vlm_prep_time = time.time() - vlm_start
        if self.cfg_whitebox_enabled:
            self.cfg_whitebox_payload['cfg_reference'] = {'mode': self.cfg_reference_mode, 'label': self.cfg_reference_label, 'instruction': self.cfg_reference_instruction, 'source_instruction': self.cfg_source_instruction, 'lambda_plus': self.cfg_lambda_plus, 'lambda_minus': self.cfg_lambda_minus, 'continuity_guard': self.cfg_continuity_guard, 'switch_window': self.cfg_switch_window, 'target_branch': self.cfg_target_branch, 'null_branch': self.cfg_null_branch, 'source_branch': self.cfg_source_branch, 'projection_mode': self.cfg_projection_mode, 'projection_strength': self.cfg_projection_strength, 'projection_prefix_steps': self.cfg_projection_prefix_steps, 'activation_window_mode': self.cfg_activation_window_mode}
            self._maybe_capture_switch_start_state()
        use_action_progress = self.progress_enabled and self.progress_feature_source.startswith('motus_action_decoder')
        if not use_action_progress:
            self._update_progress_score(vlm_inputs)
        inference_start = time.time()
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        cfg_debug_payload = None
        with torch.no_grad():
            if self.bridge_search_enabled and self.bridge_inferences_remaining > 0 and self.cfg_enabled:
                uncond_instruction = self._build_cfg_reference_instruction(scene_prefix)
                uncond_t5_out = self.t5_encoder([uncond_instruction], self.device)
                if isinstance(uncond_t5_out, torch.Tensor):
                    uncond_t5_list = [uncond_t5_out.squeeze(0)] if uncond_t5_out.dim() == 3 else [uncond_t5_out]
                elif isinstance(uncond_t5_out, list):
                    uncond_t5_list = uncond_t5_out
                else:
                    uncond_t5_list = [uncond_t5_out]
                uncond_vlm_inputs = self._preprocess_vlm_messages(uncond_instruction, uncond_first_frame_pil)
                predicted_frames, predicted_actions, bridge_window_summary = self._run_bridge_search(current_frame=current_frame, num_inference_steps=num_inference_steps, t5_list=t5_list, vlm_inputs=vlm_inputs, uncond_t5_list=uncond_t5_list, uncond_vlm_inputs=uncond_vlm_inputs, uncond_first_frame=uncond_first_frame)
                self.bridge_selection_history.append(bridge_window_summary)
                if self.switch_metadata:
                    self.switch_metadata['iti2_bridge_last_window'] = dict(bridge_window_summary)
                selected_scale = float(bridge_window_summary['selected_guidance_scale'])
                self.switch_cfg_scale_history.append(selected_scale)
                self.bridge_inferences_remaining -= 1
                if self.bridge_inferences_remaining <= 0:
                    self.bridge_search_enabled = False
                    self.cfg_enabled = False
            elif self.cfg_enabled:
                uncond_instruction = self._build_cfg_reference_instruction(scene_prefix)
                uncond_t5_out = self.t5_encoder([uncond_instruction], self.device)
                if isinstance(uncond_t5_out, torch.Tensor):
                    uncond_t5_list = [uncond_t5_out.squeeze(0)] if uncond_t5_out.dim() == 3 else [uncond_t5_out]
                elif isinstance(uncond_t5_out, list):
                    uncond_t5_list = uncond_t5_out
                else:
                    uncond_t5_list = [uncond_t5_out]
                uncond_vlm_inputs = self._preprocess_vlm_messages(uncond_instruction, uncond_first_frame_pil)
                source_t5_list = None
                source_vlm_inputs = None
                if self.cfg_reference_mode in {'dual_residual_counterfactual', 'event_triggered_source_full', 'conflict_aware_event_triggered', 'conflict_aware_sparse_source_repulsion', 'alignment_gated_source_repulsion', 'semantic_metric_source_repulsion', 'dimension_masked_source_repulsion', 'source_memory_trace_cfg', 'hysteretic_event_cfg', 'soft_sign_conflict_cfg', 'norm_balanced_cfg', 'target_novelty_boost_cfg', 'hybrid_reference_takeover_cfg', 'novelty_hybrid_takeover_cfg', 'novelty_conflict_mask_cfg'}:
                    source_instruction = self._build_cfg_source_instruction(scene_prefix)
                    source_t5_out = self.t5_encoder([source_instruction], self.device)
                    if isinstance(source_t5_out, torch.Tensor):
                        source_t5_list = [source_t5_out.squeeze(0)] if source_t5_out.dim() == 3 else [source_t5_out]
                    elif isinstance(source_t5_out, list):
                        source_t5_list = source_t5_out
                    else:
                        source_t5_list = [source_t5_out]
                    source_vlm_inputs = self._preprocess_vlm_messages(source_instruction, first_frame_pil)
                current_guidance_scale = self._get_current_cfg_scale()
                current_cfg_inference_index = self.cfg_inference_count + 1
                print(f'  [Policy] CFG: Running inference_step_cfg (scale={current_guidance_scale}, inference {self.cfg_inference_count + 1}/{self.cfg_max_inferences})', flush=True)
                if self.cfg_whitebox_enabled:
                    predicted_frames, predicted_actions, cfg_debug_payload = self.model.inference_step_cfg(first_frame=current_frame, state=self.current_state, num_inference_steps=num_inference_steps, language_embeddings=t5_list, vlm_inputs=[vlm_inputs], uncond_language_embeddings=uncond_t5_list, uncond_vlm_inputs=[uncond_vlm_inputs], uncond_first_frame=uncond_first_frame, source_language_embeddings=source_t5_list, source_vlm_inputs=[source_vlm_inputs] if source_vlm_inputs is not None else None, cfg_reference_mode=self.cfg_reference_mode, lambda_plus=self.cfg_lambda_plus, lambda_minus=self.cfg_lambda_minus, event_gate_beta=self.cfg_event_gate_beta, event_gate_threshold=self.cfg_event_gate_threshold, event_gate_stop_threshold=self.cfg_event_gate_stop_threshold, conflict_cos_threshold=self.cfg_conflict_cos_threshold, conflict_mode=self.cfg_conflict_mode, gate_mode=self.cfg_gate_mode, gate_on_threshold=self.cfg_gate_on_threshold, gate_off_threshold=self.cfg_gate_off_threshold, gate_ema_decay=self.cfg_gate_ema_decay, initial_gate_state=self.cfg_gate_state, initial_gate_ema_value=self.cfg_gate_ema_value, source_memory_decay=self.cfg_source_memory_decay, source_memory_mode=self.cfg_source_memory_mode, soft_mask_temperature=self.cfg_soft_mask_temperature, soft_mask_dominance_eta=self.cfg_soft_mask_dominance_eta, norm_balance_clip_min=self.cfg_norm_balance_clip_min, norm_balance_clip_max=self.cfg_norm_balance_clip_max, novelty_boost_scale=self.cfg_novelty_boost_scale, hybrid_reference_decay=self.cfg_hybrid_reference_decay, trust_region_enabled=self.cfg_trust_region_enabled, trust_region_coeff=self.cfg_trust_region_coeff, guidance_scale=current_guidance_scale, prior_action=self.forcing_prior_action, forcing_steps=self.forcing_steps, return_cfg_debug=True, save_cfg_vectors=self.cfg_whitebox_save_vectors)
                else:
                    predicted_frames, predicted_actions = self.model.inference_step_cfg(first_frame=current_frame, state=self.current_state, num_inference_steps=num_inference_steps, language_embeddings=t5_list, vlm_inputs=[vlm_inputs], uncond_language_embeddings=uncond_t5_list, uncond_vlm_inputs=[uncond_vlm_inputs], uncond_first_frame=uncond_first_frame, source_language_embeddings=source_t5_list, source_vlm_inputs=[source_vlm_inputs] if source_vlm_inputs is not None else None, cfg_reference_mode=self.cfg_reference_mode, lambda_plus=self.cfg_lambda_plus, lambda_minus=self.cfg_lambda_minus, event_gate_beta=self.cfg_event_gate_beta, event_gate_threshold=self.cfg_event_gate_threshold, event_gate_stop_threshold=self.cfg_event_gate_stop_threshold, conflict_cos_threshold=self.cfg_conflict_cos_threshold, conflict_mode=self.cfg_conflict_mode, gate_mode=self.cfg_gate_mode, gate_on_threshold=self.cfg_gate_on_threshold, gate_off_threshold=self.cfg_gate_off_threshold, gate_ema_decay=self.cfg_gate_ema_decay, initial_gate_state=self.cfg_gate_state, initial_gate_ema_value=self.cfg_gate_ema_value, source_memory_decay=self.cfg_source_memory_decay, source_memory_mode=self.cfg_source_memory_mode, soft_mask_temperature=self.cfg_soft_mask_temperature, soft_mask_dominance_eta=self.cfg_soft_mask_dominance_eta, norm_balance_clip_min=self.cfg_norm_balance_clip_min, norm_balance_clip_max=self.cfg_norm_balance_clip_max, novelty_boost_scale=self.cfg_novelty_boost_scale, hybrid_reference_decay=self.cfg_hybrid_reference_decay, trust_region_enabled=self.cfg_trust_region_enabled, trust_region_coeff=self.cfg_trust_region_coeff, guidance_scale=current_guidance_scale, prior_action=self.forcing_prior_action, forcing_steps=self.forcing_steps)
                self.switch_cfg_scale_history.append(float(current_guidance_scale))
                event_gate_value = None
                if cfg_debug_payload is not None:
                    raw_gate = cfg_debug_payload.get('event_gate_value')
                    if isinstance(raw_gate, (int, float)):
                        event_gate_value = float(raw_gate)
                    elif raw_gate is not None:
                        try:
                            event_gate_value = float(raw_gate)
                        except (TypeError, ValueError):
                            event_gate_value = None
                if event_gate_value is not None:
                    self.switch_event_gate_history.append(event_gate_value)
                if cfg_debug_payload is not None:
                    raw_gate_state = cfg_debug_payload.get('gate_state_final')
                    if raw_gate_state is not None:
                        self.cfg_gate_state = bool(raw_gate_state)
                        self.cfg_gate_history.append(int(self.cfg_gate_state))
                    raw_gate_ema = cfg_debug_payload.get('gate_ema_value_final')
                    if isinstance(raw_gate_ema, (int, float)):
                        self.cfg_gate_ema_value = float(raw_gate_ema)
                    if self.switch_metadata:
                        self.switch_metadata['cfg_gate_state'] = bool(self.cfg_gate_state)
                        self.switch_metadata['cfg_gate_ema_value'] = float(self.cfg_gate_ema_value)
                if self.cfg_whitebox_enabled:
                    self._append_cfg_whitebox_diagnostics(cfg_debug_payload, guidance_scale=current_guidance_scale, cfg_inference_index=current_cfg_inference_index)
                self.cfg_inference_count += 1
                if self.cfg_reference_mode in {'event_triggered_source_full', 'conflict_aware_event_triggered'}:
                    if event_gate_value is not None and event_gate_value < self.cfg_event_gate_stop_threshold:
                        print(f'  [Policy] CFG: Auto-disabled by event gate ({event_gate_value:.4f} < {self.cfg_event_gate_stop_threshold:.4f})', flush=True)
                        self.cfg_enabled = False
                        self.cfg_inference_count = 0
                        if self.switch_metadata:
                            self.switch_metadata['event_gate_auto_disabled'] = True
                            self.switch_metadata['event_gate_last_value'] = event_gate_value
                elif self.cfg_reference_mode == 'hysteretic_event_cfg':
                    if cfg_debug_payload is not None and (not bool(cfg_debug_payload.get('gate_state_final', False))):
                        print('  [Policy] CFG: Auto-disabled after hysteretic gate settled off', flush=True)
                        self.cfg_enabled = False
                        self.cfg_inference_count = 0
                        if self.switch_metadata:
                            self.switch_metadata['event_gate_auto_disabled'] = True
                            self.switch_metadata['event_gate_last_value'] = event_gate_value
                if self.cfg_inference_count >= self.cfg_max_inferences:
                    print(f'  [Policy] CFG: Auto-disabled after {self.cfg_max_inferences} inferences', flush=True)
                    self.cfg_enabled = False
                    self.cfg_inference_count = 0
            else:
                if self.forcing_prior_action is not None and self.forcing_steps > 0:
                    print(f'  [Policy] Forcing: Injecting prior action (steps={self.forcing_steps})', flush=True)
                predicted_frames, predicted_actions = self.model.inference_step(first_frame=current_frame, state=self.current_state, num_inference_steps=num_inference_steps, language_embeddings=t5_list, vlm_inputs=[vlm_inputs], prior_action=self.forcing_prior_action, forcing_steps=self.forcing_steps)
        model_inference_time = time.time() - inference_start
        if use_action_progress:
            with torch.no_grad():
                self.latest_action_progress_feature = self.model.extract_action_progress_features(first_frame=current_frame, state=self.current_state, action_sequence=predicted_actions, language_embeddings=t5_list, vlm_inputs=vlm_inputs)
            self._update_progress_score(vlm_inputs)
        self.forcing_prior_action = None
        self.forcing_steps = 0
        projection_diag_inference_index = locals().get('current_cfg_inference_index')
        save_start = time.time()
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames
                condition_frame_viz = current_frame.squeeze(0)
                predicted_frames_viz = predicted_frames_viz.squeeze(0)
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1
        save_time = time.time() - save_start
        actions_pre_projection = predicted_actions.squeeze(0).detach().float().cpu().numpy()
        actions_real = self._project_cfg_actions(actions_pre_projection)
        if self.switch_metadata:
            self.switch_metadata['cfg_projection_applied'] = bool(self.cfg_projection_mode == 'on' and self.cfg_projection_strength > 0.0)
        if self.cfg_whitebox_enabled and projection_diag_inference_index is not None:
            self._append_cfg_projection_diagnostics(actions_pre_projection, actions_real, cfg_inference_index=int(projection_diag_inference_index))
        self.prev_action = actions_real[-1].copy()
        self.last_action_chunk = actions_real.copy()
        if self.offline_trace_enabled:
            self.offline_episode_action_chunk_trace.append(actions_real.reshape(-1).tolist())
        self.action_cache.extend(actions_real)
        total_time = time.time() - start_time
        self.inference_times.append(total_time)
        self.total_inference_time += total_time
        self.inference_count += 1
        if self.step_count % 10 == 0:
            avg_time = self.total_inference_time / self.inference_count
            logger.info(f'[Timing] Step {self.step_count}: Total={total_time:.2f}s (T5={t5_time:.2f}s, VLM_prep={vlm_prep_time:.2f}s, Model={model_inference_time:.2f}s, Save={save_time:.2f}s) | Avg={avg_time:.2f}s, Freq={1 / avg_time:.3f}Hz')
        return actions_real

    def _get_current_cfg_scale(self) -> float:
        if self.cfg_schedule_scales:
            index = min(self.cfg_inference_count, len(self.cfg_schedule_scales) - 1)
            return float(self.cfg_schedule_scales[index])
        return float(self.guidance_scale)

    def _build_cfg_reference_instruction(self, scene_prefix: str) -> str:
        reference_instruction = self.cfg_reference_instruction if self.cfg_reference_mode == 'source_instruction' else self.uncond_instruction
        return f'{scene_prefix}{reference_instruction}'

    def _build_cfg_source_instruction(self, scene_prefix: str) -> str:
        return f'{scene_prefix}{self.cfg_source_instruction}'

    def _capture_bridge_source_actions(self, cached_actions_before_switch: List[np.ndarray]) -> Optional[np.ndarray]:
        if cached_actions_before_switch:
            source = np.asarray(cached_actions_before_switch, dtype=np.float32)
            if source.ndim == 1:
                return source.reshape(1, -1)
            return source
        if self.last_executed_action is not None:
            return np.asarray(self.last_executed_action, dtype=np.float32).reshape(1, -1)
        if self.last_action_chunk is not None:
            chunk = np.asarray(self.last_action_chunk, dtype=np.float32)
            if chunk.ndim == 1:
                return chunk.reshape(1, -1)
            return chunk
        return None

    def _build_bridge_source_preview(self) -> List[Dict[str, Any]]:
        if self.bridge_source_actions is None or self.bridge_source_actions.size == 0:
            return []
        preview = []
        tail = self.bridge_source_actions[-min(len(self.bridge_source_actions), 3):]
        for index, action in enumerate(tail):
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            preview.append({'source_tail_index': int(index), 'l2': float(np.linalg.norm(action_arr)), 'mean_abs': float(np.mean(np.abs(action_arr))), 'min': float(np.min(action_arr)), 'max': float(np.max(action_arr))})
        return preview

    def _build_bridge_prior_action(self) -> Optional[torch.Tensor]:
        if self.bridge_source_actions is None or self.bridge_source_actions.size == 0:
            return self.forcing_prior_action
        prior = torch.from_numpy(self.bridge_source_actions[-1]).float().unsqueeze(0).unsqueeze(0)
        return prior.to(self.device)

    def _blend_bridge_candidate(self, candidate: np.ndarray, blend_weight: float) -> np.ndarray:
        if self.bridge_source_actions is None or blend_weight <= 0.0:
            return candidate
        prefix_steps = min(self.bridge_prefix_steps, candidate.shape[0], self.bridge_source_actions.shape[0])
        if prefix_steps <= 0:
            return candidate
        source_prefix = self.bridge_source_actions[-prefix_steps:]
        blended = candidate.copy()
        blended[:prefix_steps] = blend_weight * source_prefix + (1.0 - blend_weight) * blended[:prefix_steps]
        return blended

    def _project_bridge_candidate(self, candidate: np.ndarray) -> np.ndarray:
        projected = candidate.copy()
        if self.bridge_source_actions is not None and self.bridge_projection_strength > 0.0:
            prefix_steps = min(self.bridge_prefix_steps, projected.shape[0], self.bridge_source_actions.shape[0])
            if prefix_steps > 0:
                source_prefix = self.bridge_source_actions[-prefix_steps:]
                projected[:prefix_steps] = self.bridge_projection_strength * source_prefix + (1.0 - self.bridge_projection_strength) * projected[:prefix_steps]
        action_min = self.action_min.detach().float().cpu().numpy().reshape(1, -1)
        action_max = self.action_max.detach().float().cpu().numpy().reshape(1, -1)
        return np.clip(projected, action_min, action_max).astype(np.float32, copy=False)

    def _project_cfg_actions(self, actions: np.ndarray) -> np.ndarray:
        projected = np.asarray(actions, dtype=np.float32).copy()
        if self.cfg_projection_mode != 'on' or self.cfg_projection_strength <= 0.0:
            return projected
        if self.bridge_source_actions is not None and self.bridge_source_actions.size > 0:
            prefix_steps = min(self.cfg_projection_prefix_steps, projected.shape[0], self.bridge_source_actions.shape[0])
            if prefix_steps > 0:
                source_prefix = self.bridge_source_actions[-prefix_steps:]
                projected[:prefix_steps] = self.cfg_projection_strength * source_prefix + (1.0 - self.cfg_projection_strength) * projected[:prefix_steps]
        action_min = self.action_min.detach().float().cpu().numpy().reshape(1, -1)
        action_max = self.action_max.detach().float().cpu().numpy().reshape(1, -1)
        return np.clip(projected, action_min, action_max).astype(np.float32, copy=False)

    def _bridge_prefix_rmse(self, candidate: np.ndarray, reference: np.ndarray) -> float:
        prefix_steps = min(self.bridge_prefix_steps, candidate.shape[0], reference.shape[0])
        if prefix_steps <= 0:
            return 0.0
        delta = candidate[:prefix_steps] - reference[-prefix_steps:]
        return float(np.sqrt(np.mean(np.square(delta))))

    def _bridge_boundary_cost(self, candidate: np.ndarray) -> float:
        if self.bridge_source_actions is None or self.bridge_source_actions.size == 0:
            return 0.0
        first_action = candidate[0]
        source_last = self.bridge_source_actions[-1]
        first_step_l2 = float(np.linalg.norm(first_action - source_last))
        jerk_cost = 0.0
        if candidate.shape[0] >= 2:
            candidate_delta = candidate[1] - candidate[0]
            if self.bridge_source_actions.shape[0] >= 2:
                source_delta = self.bridge_source_actions[-1] - self.bridge_source_actions[-2]
            else:
                source_delta = np.zeros_like(candidate_delta)
            jerk_cost = float(np.linalg.norm(candidate_delta - source_delta))
        return first_step_l2 + 0.5 * jerk_cost

    def _bridge_target_alignment(self, candidate: np.ndarray, anchor: np.ndarray) -> float:
        count = min(candidate.shape[0], anchor.shape[0])
        if count <= 0:
            return 0.0
        rmse = float(np.sqrt(np.mean(np.square(candidate[:count] - anchor[:count]))))
        return 1.0 / (1.0 + rmse)

    def _bridge_uncertainty_cost(self, candidate: np.ndarray, ensemble_mean: np.ndarray) -> float:
        count = min(candidate.shape[0], ensemble_mean.shape[0])
        if count <= 0:
            return 0.0
        return float(np.mean(np.abs(candidate[:count] - ensemble_mean[:count])))

    def _summarize_bridge_candidate(self, action_array: np.ndarray) -> Dict[str, Any]:
        action_array = np.asarray(action_array, dtype=np.float32)
        return {'shape': [int(action_array.shape[0]), int(action_array.shape[1])], 'mean_abs': float(np.mean(np.abs(action_array))), 'max_abs': float(np.max(np.abs(action_array))), 'mean_l2': float(np.mean(np.linalg.norm(action_array, axis=1)))}

    def _run_bridge_search(self, current_frame: torch.Tensor, num_inference_steps: int, t5_list: List[torch.Tensor], vlm_inputs: Dict[str, torch.Tensor], uncond_t5_list: List[torch.Tensor], uncond_vlm_inputs: Dict[str, torch.Tensor], uncond_first_frame: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        candidates = []
        bridge_prior_action = self._build_bridge_prior_action()
        forcing_steps = max(self.forcing_steps, 1 if bridge_prior_action is not None else 0)
        for index, guidance_scale in enumerate(self.bridge_candidate_scales):
            predicted_frames, predicted_actions = self.model.inference_step_cfg(first_frame=current_frame, state=self.current_state, num_inference_steps=num_inference_steps, language_embeddings=t5_list, vlm_inputs=[vlm_inputs], uncond_language_embeddings=uncond_t5_list, uncond_vlm_inputs=[uncond_vlm_inputs], uncond_first_frame=uncond_first_frame, guidance_scale=float(guidance_scale), prior_action=bridge_prior_action, forcing_steps=forcing_steps)
            candidate_actions = predicted_actions.squeeze(0).detach().float().cpu().numpy()
            blended_actions = self._blend_bridge_candidate(candidate_actions, self.bridge_blend_weights[index] if index < len(self.bridge_blend_weights) else 0.0)
            projected_actions = self._project_bridge_candidate(blended_actions)
            candidates.append({'index': int(index), 'guidance_scale': float(guidance_scale), 'blend_weight': float(self.bridge_blend_weights[index]) if index < len(self.bridge_blend_weights) else 0.0, 'predicted_frames': predicted_frames, 'tensor_actions': torch.from_numpy(projected_actions).unsqueeze(0).to(predicted_actions.device, dtype=predicted_actions.dtype), 'projected_actions': projected_actions})
        anchor_index = max(range(len(candidates)), key=lambda idx: candidates[idx]['guidance_scale'])
        target_anchor = candidates[anchor_index]['projected_actions']
        ensemble_mean = np.mean([candidate['projected_actions'] for candidate in candidates], axis=0)
        for candidate in candidates:
            projected_actions = candidate['projected_actions']
            source_guard_cost = self._bridge_prefix_rmse(projected_actions, self.bridge_source_actions) if self.bridge_source_actions is not None else 0.0
            boundary_cost = self._bridge_boundary_cost(projected_actions)
            target_alignment = self._bridge_target_alignment(projected_actions, target_anchor)
            uncertainty_cost = self._bridge_uncertainty_cost(projected_actions, ensemble_mean)
            score = self.bridge_target_weight * target_alignment - self.bridge_source_guard_weight * source_guard_cost - self.bridge_boundary_weight * boundary_cost - self.bridge_uncertainty_weight * uncertainty_cost
            candidate['summary'] = {'candidate_index': candidate['index'], 'guidance_scale': candidate['guidance_scale'], 'blend_weight': candidate['blend_weight'], 'target_alignment': target_alignment, 'source_guard_cost': source_guard_cost, 'boundary_cost': boundary_cost, 'uncertainty_cost': uncertainty_cost, 'score': float(score), 'action_stats': self._summarize_bridge_candidate(projected_actions)}
        selected_index = max(range(len(candidates)), key=lambda idx: candidates[idx]['summary']['score'])
        selected = candidates[selected_index]
        window_summary = {'window_index': len(self.bridge_selection_history) + 1, 'selected_candidate_index': int(selected_index), 'selected_guidance_scale': float(selected['guidance_scale']), 'target_anchor_index': int(anchor_index), 'bridge_inferences_remaining_after_this': max(self.bridge_inferences_remaining - 1, 0), 'candidate_count': len(candidates), 'source_action_count': 0 if self.bridge_source_actions is None else int(self.bridge_source_actions.shape[0]), 'candidate_summaries': [candidate['summary'] for candidate in candidates]}
        return (selected['predicted_frames'], selected['tensor_actions'], window_summary)

    def _maybe_capture_switch_start_state(self) -> None:
        if not self.cfg_whitebox_enabled or 'switch_start_state' in self.cfg_whitebox_payload:
            return
        if self.current_state is None:
            return
        state = self.current_state.detach().float().cpu().numpy().reshape(-1)
        state_summary = {'dim': int(state.size), 'l2': float(np.linalg.norm(state)), 'min': float(np.min(state)), 'max': float(np.max(state)), 'mean': float(np.mean(state)), 'abs_max': float(np.max(np.abs(state)))}
        if self.current_state_norm is not None:
            state_norm = self.current_state_norm.detach().float().cpu().numpy().reshape(-1)
            below_zero = np.clip(-state_norm, 0.0, None)
            above_one = np.clip(state_norm - 1.0, 0.0, None)
            overflow = below_zero + above_one
            state_summary['normalized_expected_range'] = {'min': float(np.min(state_norm)), 'max': float(np.max(state_norm)), 'below_zero_count': int(np.count_nonzero(below_zero)), 'above_one_count': int(np.count_nonzero(above_one)), 'out_of_range_count': int(np.count_nonzero(overflow)), 'overflow_linf': float(np.max(overflow)) if overflow.size else 0.0, 'overflow_l1_mean': float(np.mean(overflow)) if overflow.size else 0.0}
            if self.cfg_whitebox_save_vectors:
                state_summary['state_vector_normalized'] = state_norm.tolist()
        if self.cfg_whitebox_save_vectors:
            state_summary['state_vector'] = state.tolist()
        self.cfg_whitebox_payload['switch_start_state'] = state_summary

    def _append_cfg_whitebox_diagnostics(self, cfg_debug_payload: Optional[Dict[str, torch.Tensor]], guidance_scale: float, cfg_inference_index: int) -> None:
        if not self.cfg_whitebox_enabled:
            return
        diagnostics = self.cfg_whitebox_payload.setdefault('cfg_inference_diagnostics', [])
        if not cfg_debug_payload:
            diagnostics.append({'cfg_inference_index': int(cfg_inference_index), 'guidance_scale': float(guidance_scale), 'warning': 'cfg_debug_payload_missing'})
            return
        action_final = self._cfg_tensor_to_action_array(cfg_debug_payload.get('predicted_actions_final'))
        action_target = self._cfg_tensor_to_action_array(cfg_debug_payload.get('predicted_actions_target', cfg_debug_payload.get('predicted_actions_cond')))
        action_null = self._cfg_tensor_to_action_array(cfg_debug_payload.get('predicted_actions_null', cfg_debug_payload.get('predicted_actions_uncond')))
        action_source = self._cfg_tensor_to_action_array(cfg_debug_payload.get('predicted_actions_source'))
        branch_payloads = {'target': self._build_cf_branch_payload(action_target, self.cfg_target_branch, instruction=self.current_instruction), 'null': self._build_cf_branch_payload(action_null, self.cfg_null_branch, instruction=self.cfg_reference_instruction), 'source': self._build_cf_branch_payload(action_source, self.cfg_source_branch, instruction=self.cfg_source_instruction)}
        diagnostic = {'cfg_inference_index': int(cfg_inference_index), 'guidance_scale': float(guidance_scale), 'cfg_reference_mode': str(cfg_debug_payload.get('cfg_reference_mode', self.cfg_reference_mode)), 'prediction_type': str(cfg_debug_payload.get('prediction_type', 'flow_matching_velocity')), 'scheduler_family': str(cfg_debug_payload.get('scheduler_family', 'manual_euler_flow_matching')), 'lambda_plus': float(cfg_debug_payload.get('lambda_plus', self.cfg_lambda_plus)), 'lambda_minus': float(cfg_debug_payload.get('lambda_minus', self.cfg_lambda_minus)), 'cf_branch_payloads': branch_payloads, 'action_final': self._build_action_diagnostics(action_final), 'action_target': self._build_action_diagnostics(action_target), 'action_null': self._build_action_diagnostics(action_null)}
        event_gate_value = cfg_debug_payload.get('event_gate_value')
        if event_gate_value is not None:
            diagnostic['event_gate_value'] = float(event_gate_value)
        disagreement_norm = cfg_debug_payload.get('event_gate_disagreement_norm')
        if disagreement_norm is not None:
            diagnostic['event_gate_disagreement_norm'] = float(disagreement_norm)
        conflict_cosine = cfg_debug_payload.get('conflict_cosine')
        if conflict_cosine is not None:
            diagnostic['conflict_cosine'] = float(conflict_cosine)
        conflict_fraction = cfg_debug_payload.get('conflict_fraction')
        if conflict_fraction is not None:
            diagnostic['conflict_fraction'] = float(conflict_fraction)
        diagnostic['action_cond'] = dict(diagnostic['action_target'])
        diagnostic['action_uncond'] = dict(diagnostic['action_null'])
        target_null_delta = self._build_action_delta_diagnostics(action_target, action_null)
        if target_null_delta is not None:
            diagnostic['target_null_delta'] = target_null_delta
            diagnostic['cond_uncond_delta'] = dict(target_null_delta)
        source_null_delta = self._build_action_delta_diagnostics(action_source, action_null)
        if source_null_delta is not None:
            diagnostic['source_null_delta'] = source_null_delta
        target_source_delta = self._build_action_delta_diagnostics(action_target, action_source)
        if target_source_delta is not None:
            diagnostic['target_source_delta'] = target_source_delta
        if action_source is not None:
            diagnostic['action_source'] = self._build_action_diagnostics(action_source)
        if self.cfg_whitebox_save_vectors:
            diagnostic['action_final_vector'] = action_final.tolist() if action_final is not None else []
            diagnostic['action_target_vector'] = action_target.tolist() if action_target is not None else []
            diagnostic['action_null_vector'] = action_null.tolist() if action_null is not None else []
            diagnostic['action_cond_vector'] = list(diagnostic['action_target_vector'])
            diagnostic['action_uncond_vector'] = list(diagnostic['action_null_vector'])
            if action_source is not None:
                diagnostic['action_source_vector'] = action_source.tolist()
        denoise_step_payloads = cfg_debug_payload.get('denoise_step_payloads')
        if denoise_step_payloads:
            diagnostic['denoise_step_payloads'] = list(denoise_step_payloads)
        self.cfg_whitebox_payload['trace_schema_version'] = 'noise_space_v1'
        self.cfg_whitebox_payload['cf_branch_payloads'] = branch_payloads
        if self.switch_metadata:
            self.switch_metadata['cf_branch_payloads'] = dict(branch_payloads)
        diagnostics.append(diagnostic)

    def _append_cfg_projection_diagnostics(self, pre_projection_actions: np.ndarray, post_projection_actions: np.ndarray, cfg_inference_index: int) -> None:
        if not self.cfg_whitebox_enabled:
            return
        diagnostics = self.cfg_whitebox_payload.setdefault('cfg_projection_diagnostics', [])
        diagnostic = {'cfg_inference_index': int(cfg_inference_index), 'projection_mode': str(self.cfg_projection_mode), 'projection_strength': float(self.cfg_projection_strength), 'projection_prefix_steps': int(self.cfg_projection_prefix_steps), 'projection_applied': bool(self.cfg_projection_mode == 'on' and self.cfg_projection_strength > 0.0), 'action_pre_projection': self._build_action_diagnostics(pre_projection_actions), 'action_post_projection': self._build_action_diagnostics(post_projection_actions)}
        projection_delta = self._build_action_delta_diagnostics(post_projection_actions, pre_projection_actions)
        if projection_delta is not None:
            diagnostic['projection_delta'] = projection_delta
        if self.cfg_whitebox_save_vectors:
            diagnostic['action_pre_projection_vector'] = np.asarray(pre_projection_actions, dtype=np.float32).tolist()
            diagnostic['action_post_projection_vector'] = np.asarray(post_projection_actions, dtype=np.float32).tolist()
        self.cfg_whitebox_payload['trace_schema_version'] = 'noise_space_v1'
        diagnostics.append(diagnostic)

    def _cfg_tensor_to_action_array(self, tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if tensor is None:
            return None
        action_dim = int(tensor.shape[-1])
        return tensor.detach().float().cpu().numpy().reshape(-1, action_dim)

    def _build_cf_branch_payload(self, action_array: Optional[np.ndarray], branch_label: str, instruction: str) -> Dict[str, Any]:
        payload = {'label': str(branch_label or ''), 'instruction_present': bool(str(instruction or '').strip()), 'non_empty': bool(action_array is not None and action_array.size > 0)}
        if action_array is None or action_array.size == 0:
            return payload
        diagnostics = self._build_action_diagnostics(action_array)
        payload.update({'shape': diagnostics.get('shape'), 'l2_mean': diagnostics.get('l2_mean'), 'abs_max': diagnostics.get('abs_max')})
        return payload

    def _build_action_diagnostics(self, action_array: np.ndarray) -> Dict[str, Any]:
        if action_array is None:
            return {}
        action_array = np.asarray(action_array, dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        else:
            action_array = action_array.reshape(-1, action_array.shape[-1])
        if action_array.size == 0:
            return {}
        step_l2 = np.linalg.norm(action_array, axis=1)
        action_min = self.action_min.detach().float().cpu().numpy().reshape(1, -1)
        action_max = self.action_max.detach().float().cpu().numpy().reshape(1, -1)
        action_range = np.maximum(action_max - action_min, 1e-06)
        normalized = (action_array - action_min) / action_range
        return {'shape': [int(action_array.shape[0]), int(action_array.shape[1])], 'l2_mean': float(np.mean(step_l2)), 'l2_max': float(np.max(step_l2)), 'min': float(np.min(action_array)), 'max': float(np.max(action_array)), 'abs_max': float(np.max(np.abs(action_array))), 'normalized_expected_range': self._compute_expected_range_stats(normalized, 0.0, 1.0), 'dataset_denorm_range': self._compute_expected_range_stats(action_array, action_min, action_max)}

    def _build_action_sequence_magnitude_diagnostics(self, action_array: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if action_array is None:
            return None
        action_array = np.asarray(action_array, dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        else:
            action_array = action_array.reshape(-1, action_array.shape[-1])
        if action_array.size == 0 or action_array.shape[0] == 0:
            return None
        step_l2 = np.linalg.norm(action_array, axis=1)
        l2_first = float(step_l2[0])
        l2_last = float(step_l2[-1])
        l2_delta = float(l2_last - l2_first)
        denom = max(action_array.shape[0] - 1, 1)
        return {'count': int(action_array.shape[0]), 'l2_first': l2_first, 'l2_last': l2_last, 'l2_min': float(np.min(step_l2)), 'l2_max': float(np.max(step_l2)), 'l2_mean': float(np.mean(step_l2)), 'l2_delta': l2_delta, 'l2_slope': float(l2_delta / denom)}

    def _build_action_delta_diagnostics(self, lhs_action_array: Optional[np.ndarray], rhs_action_array: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if lhs_action_array is None or rhs_action_array is None:
            return None
        if lhs_action_array.size == 0 or rhs_action_array.size == 0:
            return None
        count = min(lhs_action_array.shape[0], rhs_action_array.shape[0])
        if count <= 0:
            return None
        delta = lhs_action_array[:count] - rhs_action_array[:count]
        abs_delta = np.abs(delta)
        return {'delta_l2': float(np.linalg.norm(delta.reshape(-1))), 'delta_l1_mean': float(np.mean(abs_delta)), 'delta_linf': float(np.max(abs_delta)) if abs_delta.size else 0.0, 'all_but_zero': bool(np.all(abs_delta <= self.cfg_whitebox_zero_eps)), 'all_but_zero_eps': float(self.cfg_whitebox_zero_eps)}

    def _build_action_pair_cosine_diagnostics(self, lhs_action_array: Optional[np.ndarray], rhs_action_array: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if lhs_action_array is None or rhs_action_array is None:
            return None
        if lhs_action_array.size == 0 or rhs_action_array.size == 0:
            return None
        lhs = np.asarray(lhs_action_array, dtype=np.float32)
        rhs = np.asarray(rhs_action_array, dtype=np.float32)
        if lhs.ndim == 1:
            lhs = lhs.reshape(1, -1)
        if rhs.ndim == 1:
            rhs = rhs.reshape(1, -1)
        count = min(lhs.shape[0], rhs.shape[0])
        if count <= 0:
            return None
        cosines = []
        for idx in range(count):
            lhs_step = lhs[idx].reshape(-1)
            rhs_step = rhs[idx].reshape(-1)
            lhs_norm = float(np.linalg.norm(lhs_step))
            rhs_norm = float(np.linalg.norm(rhs_step))
            if lhs_norm <= 1e-08 or rhs_norm <= 1e-08:
                cosines.append(0.0)
                continue
            cosines.append(float(np.dot(lhs_step, rhs_step) / (lhs_norm * rhs_norm)))
        if not cosines:
            return None
        return {'mean': float(np.mean(cosines)), 'min': float(np.min(cosines)), 'std': float(np.std(cosines)), 'count': int(len(cosines))}

    def _compute_action_sequence_cosine_stats(self, action_array: np.ndarray) -> Dict[str, Any]:
        if action_array is None:
            return {'mean': 0.0, 'min': 0.0, 'std': 0.0, 'count': 0}
        sequence = np.asarray(action_array, dtype=np.float32)
        if sequence.ndim == 1:
            sequence = sequence.reshape(1, -1)
        if sequence.shape[0] <= 1:
            return {'mean': 1.0, 'min': 1.0, 'std': 0.0, 'count': 0}
        cosines = []
        for idx in range(sequence.shape[0] - 1):
            lhs = sequence[idx].reshape(-1)
            rhs = sequence[idx + 1].reshape(-1)
            lhs_norm = float(np.linalg.norm(lhs))
            rhs_norm = float(np.linalg.norm(rhs))
            if lhs_norm <= 1e-08 or rhs_norm <= 1e-08:
                cosines.append(0.0)
                continue
            cosines.append(float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm)))
        if not cosines:
            return {'mean': 1.0, 'min': 1.0, 'std': 0.0, 'count': 0}
        return {'mean': float(np.mean(cosines)), 'min': float(np.min(cosines)), 'std': float(np.std(cosines)), 'count': int(len(cosines))}

    def _compute_boundary_cost_against_source(self, candidate: Optional[np.ndarray], source_actions: Optional[np.ndarray]) -> float:
        if candidate is None or source_actions is None or source_actions.size == 0:
            return 0.0
        candidate_arr = np.asarray(candidate, dtype=np.float32)
        if candidate_arr.ndim == 1:
            candidate_arr = candidate_arr.reshape(1, -1)
        source_arr = np.asarray(source_actions, dtype=np.float32)
        if source_arr.ndim == 1:
            source_arr = source_arr.reshape(1, -1)
        first_action = candidate_arr[0]
        source_last = source_arr[-1]
        first_step_l2 = float(np.linalg.norm(first_action - source_last))
        jerk_cost = 0.0
        if candidate_arr.shape[0] >= 2:
            candidate_delta = candidate_arr[1] - candidate_arr[0]
            if source_arr.shape[0] >= 2:
                source_delta = source_arr[-1] - source_arr[-2]
            else:
                source_delta = np.zeros_like(candidate_delta)
            jerk_cost = float(np.linalg.norm(candidate_delta - source_delta))
        return first_step_l2 + 0.5 * jerk_cost

    def _compute_expected_range_stats(self, values: np.ndarray, lower_bound: float | np.ndarray, upper_bound: float | np.ndarray) -> Dict[str, Any]:
        values = np.asarray(values, dtype=np.float32)
        lower = np.asarray(lower_bound, dtype=np.float32)
        upper = np.asarray(upper_bound, dtype=np.float32)
        below = np.clip(lower - values, 0.0, None)
        above = np.clip(values - upper, 0.0, None)
        overflow = below + above
        return {'min': float(np.min(values)), 'max': float(np.max(values)), 'below_bound_count': int(np.count_nonzero(below)), 'above_bound_count': int(np.count_nonzero(above)), 'out_of_range_count': int(np.count_nonzero(overflow)), 'overflow_linf': float(np.max(overflow)) if overflow.size else 0.0, 'overflow_l1_mean': float(np.mean(overflow)) if overflow.size else 0.0}

    def _build_post_switch_action_preview(self) -> List[Dict[str, Any]]:
        if self.cfg_whitebox_first_k <= 0 or not self.post_switch_action_trace:
            return []
        preview = []
        for index, action in enumerate(self.post_switch_action_trace[:self.cfg_whitebox_first_k]):
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            item = {'post_switch_action_index': int(index), 'l2': float(np.linalg.norm(action_arr)), 'mean_abs': float(np.mean(np.abs(action_arr))), 'min': float(np.min(action_arr)), 'max': float(np.max(action_arr))}
            if self.cfg_whitebox_save_vectors:
                item['vector'] = action_arr.tolist()
            preview.append(item)
        return preview

    def _normalize_uncond_visual_mode(self, mode: str) -> str:
        resolved = str(mode or 'clear').strip().lower()
        if resolved in {'clear', 'none', 'identity'}:
            return 'clear'
        if resolved in {'blur', 'blurred', 'gaussian', 'gaussian_blur'}:
            return 'blurred'
        raise ValueError(f'Unknown uncond_visual_mode: {mode}')

    def _normalize_cfg_reference_mode(self, mode: str) -> str:
        resolved = str(mode or 'runtime_uncond').strip().lower()
        if resolved in {'runtime_uncond', 'source_instruction', 'dual_residual_counterfactual', 'event_triggered_source_full', 'conflict_aware_event_triggered', 'conflict_aware_sparse_source_repulsion', 'alignment_gated_source_repulsion', 'semantic_metric_source_repulsion', 'dimension_masked_source_repulsion', 'source_memory_trace_cfg', 'hysteretic_event_cfg', 'soft_sign_conflict_cfg', 'norm_balanced_cfg', 'target_novelty_boost_cfg', 'hybrid_reference_takeover_cfg', 'novelty_hybrid_takeover_cfg', 'novelty_conflict_mask_cfg'}:
            return resolved
        raise ValueError(f'Unknown cfg_reference_mode: {mode}')

    def _normalize_continuity_guard(self, mode: str) -> str:
        resolved = str(mode or 'off').strip().lower()
        if resolved in {'off', 'on', 'native_track_behavior'}:
            return resolved
        return resolved

    def _normalize_cfg_projection_mode(self, mode: str) -> str:
        resolved = str(mode or 'off').strip().lower()
        if resolved in {'off', 'on'}:
            return resolved
        raise ValueError(f'Unknown cfg_projection_mode: {mode}')

    def _normalize_blur_kernel_size(self, kernel_size: int) -> int:
        resolved = int(kernel_size or 0)
        if resolved <= 0:
            return 0
        if resolved % 2 == 0:
            resolved += 1
        return resolved

    def _compute_trace_stats(self) -> Dict[str, Any]:
        return self._compute_trace_stats_from_trace(self.post_switch_action_trace)

    def _compute_trace_stats_from_trace(self, action_trace: List[List[float]]) -> Dict[str, Any]:
        if not action_trace:
            return {}
        action_array = np.asarray(action_trace, dtype=np.float32)
        flat = action_array.reshape(-1)
        step_deltas = np.diff(action_array, axis=0) if action_array.shape[0] >= 2 else None
        stats = {}
        if 'mean_abs_all' in self.trace_stat_keys:
            stats['mean_abs_all'] = float(np.mean(np.abs(flat)))
        if 'std_all' in self.trace_stat_keys:
            stats['std_all'] = float(np.std(flat))
        if 'mean_l2' in self.trace_stat_keys:
            stats['mean_l2'] = float(np.mean(np.linalg.norm(action_array, axis=1)))
        if 'first_step_l2_disp' in self.trace_stat_keys:
            if action_array.shape[0] >= 2:
                stats['first_step_l2_disp'] = float(np.linalg.norm(action_array[1] - action_array[0]))
            else:
                stats['first_step_l2_disp'] = 0.0
        if 'step_delta_l2_mean' in self.trace_stat_keys:
            if step_deltas is not None and step_deltas.size:
                stats['step_delta_l2_mean'] = float(np.mean(np.linalg.norm(step_deltas, axis=1)))
            else:
                stats['step_delta_l2_mean'] = 0.0
        if 'action_jerk_proxy' in self.trace_stat_keys:
            if action_array.shape[0] >= 3:
                jerk = np.diff(action_array, axis=0, n=2)
                stats['action_jerk_proxy'] = float(np.mean(np.linalg.norm(jerk, axis=1)))
            else:
                stats['action_jerk_proxy'] = 0.0
        finite_mask = np.isfinite(action_array)
        stats['all_finite'] = bool(np.all(finite_mask))
        stats['has_nan'] = bool(np.isnan(action_array).any())
        stats['has_inf'] = bool(np.isinf(action_array).any())
        stats['numerical_stability_violation_count'] = int(action_array.size - np.count_nonzero(finite_mask))
        if np.any(finite_mask):
            stats['max_abs_all'] = float(np.max(np.abs(action_array[finite_mask])))
        else:
            stats['max_abs_all'] = 0.0
        return stats

    def _build_offline_trace_payload(self) -> Optional[Dict[str, Any]]:
        if not self.offline_trace_enabled:
            return None
        if self.offline_episode_action_chunk_trace:
            action_trace = list(self.offline_episode_action_chunk_trace)
            trace_unit = 'action_chunk_flat'
        elif self.offline_episode_action_trace:
            action_trace = list(self.offline_episode_action_trace)
            trace_unit = 'executed_action'
        else:
            return None
        mode = str(self.offline_trace_mode or 'full').strip().lower()
        window = int(self.offline_trace_window or 0)
        if mode in {'head', 'head_k', 'entry', 'entry_head', 's2_entry'}:
            selected = action_trace[:window] if window > 0 else action_trace
            start_index = 0
        elif mode in {'tail', 'tail_k', 'exit', 'exit_tail', 's1_tail'}:
            selected = action_trace[-window:] if window > 0 else action_trace
            start_index = max(len(action_trace) - len(selected), 0)
        else:
            selected = action_trace
            start_index = 0
        if not selected:
            return None
        end_index = start_index + len(selected) - 1
        metadata = {'corpus_role': str(self.offline_corpus_role or ''), 'trace_mode': mode, 'trace_window': window, 'selected_start_index': int(start_index), 'selected_end_index': int(end_index), 'total_action_count': int(len(action_trace)), 'selected_action_count': int(len(selected)), 'space_name': str(self.offline_space_name or 'action_chunk'), 'export_tag': str(self.offline_export_tag or ''), 'trace_unit': trace_unit}
        return {'metadata': metadata, 'trace_stats': self._compute_trace_stats_from_trace(selected), 'action_trace': selected}

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert('RGB')
        np_img = np.array(image, dtype=np.float32, copy=True) / 255.0
        return torch.from_numpy(np_img).permute(2, 0, 1)

    def _apply_gaussian_blur(self, image: Image.Image, kernel_size: int, sigma: float) -> Image.Image:
        resolved_kernel = self._normalize_blur_kernel_size(kernel_size)
        if resolved_kernel <= 1:
            return image.copy()
        rgb_image = image.convert('RGB')
        image_np = np.asarray(rgb_image)
        blurred_np = cv2.GaussianBlur(image_np, (resolved_kernel, resolved_kernel), sigmaX=float(sigma), sigmaY=float(sigma))
        return Image.fromarray(blurred_np.astype(np.uint8), mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': instruction}, {'type': 'image', 'image': image}]}]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {'input_ids': encoded['input_ids'].to(self.device), 'attention_mask': encoded['attention_mask'].to(self.device), 'pixel_values': encoded['pixel_values'].to(self.device), 'image_grid_thw': encoded.get('image_grid_thw', None)}
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        try:
            stat_path = Path(__file__).parent / 'utils' / 'stat.json'
            with open(stat_path, 'r') as f:
                stat_data = yaml.safe_load(f) if stat_path.suffix in ['.yml', '.yaml'] else None
        except Exception:
            stat_data = None
        if stat_data is None:
            import json as _json
            with open(Path(__file__).parent / 'utils' / 'stat.json', 'r') as f:
                stat_data = _json.load(f)
        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats not found')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)

    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:

        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)
        return Image.fromarray(grid_image)

    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        if not self.save_images:
            return
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f'episode_{self.episode_count:04d}_step_{self.step_count:04d}.png'
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f'Saved frame grid to {save_path}')
        except Exception as e:
            logger.warning(f'Failed to save frame grid: {e}')

    def print_timing_stats(self):
        if self.inference_count == 0:
            logger.info('No inference timing data available')
            return
        avg_time = self.total_inference_time / self.inference_count
        min_time = min(self.inference_times)
        max_time = max(self.inference_times)
        logger.info('=' * 80)
        logger.info('INFERENCE TIMING STATISTICS')
        logger.info('=' * 80)
        logger.info(f'Total inferences: {self.inference_count}')
        logger.info(f'Total time: {self.total_inference_time:.2f}s')
        logger.info(f'Average time per inference: {avg_time:.2f}s')
        logger.info(f'Min time: {min_time:.2f}s')
        logger.info(f'Max time: {max_time:.2f}s')
        logger.info(f'Average inference frequency: {1 / avg_time:.3f} Hz')
        logger.info(f'Estimated time per step (real-world): {avg_time:.2f}s')
        logger.info('=' * 80)

def encode_obs(observation):
    return observation

def get_model(usr_args):
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')
    vlm_path = usr_args.get('vlm_path')
    if not wan_path:
        raise ValueError('wan_path not provided in usr_args')
    if not vlm_path:
        raise ValueError('vlm_path not provided in usr_args')
    policy_dir = Path(__file__).parent
    config_path = policy_dir / 'utils' / 'robotwin.yml'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = MotusPolicy(checkpoint_path=checkpoint_path, wan_path=wan_path, vlm_path=vlm_path, config_path=str(config_path), device=device, log_dir=usr_args.get('log_dir'), task_name=usr_args.get('task_name'), progress_checkpoint_path=usr_args.get('progress_checkpoint'))
    return policy

def eval(TASK_ENV, model, observation):
    if len(model.action_cache) > 0:
        action = model.action_cache.popleft()
    else:
        obs = encode_obs(observation)
        instruction = TASK_ENV.get_instruction()
        model.set_instruction(instruction)
        model.update_obs(obs)
        actions = model.get_action()
        action = actions[0]
        if len(actions) > 1:
            model.action_cache.extend(actions[1:])
    if hasattr(model, 'record_executed_action'):
        model.record_executed_action(action)
    TASK_ENV.take_action(action, action_type='qpos')

def reset_model(model):
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.forcing_steps = 0
    model.forcing_prior_action = None
    model.cfg_enabled = False
    model.cfg_inference_count = 0
    model.guidance_scale = 2.0
    model.cfg_schedule_name = 'constant'
    model.cfg_schedule_scales = []
    model.cfg_reference_mode = 'runtime_uncond'
    model.cfg_reference_label = 'runtime_uncond'
    model.cfg_reference_instruction = ''
    model.cfg_source_instruction = ''
    model.cfg_lambda_plus = 1.0
    model.cfg_lambda_minus = 0.0
    model.cfg_continuity_guard = 'off'
    model.cfg_switch_window = 0
    model.cfg_target_branch = 'target'
    model.cfg_null_branch = 'null'
    model.cfg_source_branch = 'source'
    model.cfg_projection_mode = 'off'
    model.cfg_projection_strength = 0.0
    model.cfg_projection_prefix_steps = 2
    model.cfg_activation_window_mode = ''
    model.uncond_instruction = ''
    model.uncond_visual_mode = 'clear'
    model.uncond_blur_kernel_size = 0
    model.uncond_blur_sigma = 0.0
    model.switch_trace_enabled = False
    model.trace_stat_keys = ['mean_abs_all', 'std_all', 'mean_l2']
    model.switch_metadata = {}
    model.switch_cfg_scale_history = []
    model.post_switch_action_trace = []
    model.cfg_whitebox_enabled = False
    model.cfg_whitebox_save_vectors = False
    model.cfg_whitebox_first_k = 5
    model.cfg_whitebox_zero_eps = 0.001
    model.cfg_whitebox_payload = {}
    model.last_executed_action = None
    model.last_action_chunk = None
    model.offline_trace_enabled = False
    model.offline_corpus_role = ''
    model.offline_trace_mode = 'off'
    model.offline_trace_window = 0
    model.offline_space_name = 'action_chunk'
    model.offline_export_tag = ''
    model.offline_episode_action_trace = []
    model.offline_episode_action_chunk_trace = []
    model.bridge_search_enabled = False
    model.bridge_inferences_remaining = 0
    model.bridge_candidate_scales = []
    model.bridge_blend_weights = []
    model.bridge_prefix_steps = 2
    model.bridge_projection_strength = 0.25
    model.bridge_target_weight = 1.0
    model.bridge_source_guard_weight = 0.3
    model.bridge_boundary_weight = 0.2
    model.bridge_uncertainty_weight = 0.1
    model.bridge_source_actions = None
    model.bridge_selection_history = []
    if hasattr(model, '_reset_progress_state'):
        model._reset_progress_state()
    model.episode_count += 1
    model.step_count = 0
    logger.info(f'Model reset completed for episode {model.episode_count}')
