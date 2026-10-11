import torch
import numpy as np
import cv2
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

    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, vlm_path: str, device: str='cuda', log_dir: Optional[str]=None, task_name: Optional[str]=None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
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
        self.current_instruction = ''
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
        logger.info('Motus Policy initialized successfully')

    def set_instruction(self, instruction: str):
        self.current_instruction = instruction
        logger.info(f'Instruction set: {instruction}')

    def prepare_instruction_switch(self, clear_action_cache: bool=True) -> Dict[str, Any]:
        cache_len = len(self.action_cache)
        if clear_action_cache:
            self.action_cache.clear()
        self.obs_cache.clear()
        self.current_instruction = ''
        return {'cache_len_before_switch': cache_len,
                'cache_cleared': bool(clear_action_cache)}

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
        vlm_prep_time = time.time() - vlm_start
        inference_start = time.time()
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(first_frame=current_frame, state=self.current_state, num_inference_steps=num_inference_steps, language_embeddings=t5_list, vlm_inputs=[vlm_inputs])
        model_inference_time = time.time() - inference_start
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
        actions_real = predicted_actions.squeeze(0).detach().float().cpu().numpy()
        total_time = time.time() - start_time
        self.inference_times.append(total_time)
        self.total_inference_time += total_time
        self.inference_count += 1
        if self.step_count % 10 == 0:
            avg_time = self.total_inference_time / self.inference_count
            logger.info(f'[Timing] Step {self.step_count}: Total={total_time:.2f}s (T5={t5_time:.2f}s, VLM_prep={vlm_prep_time:.2f}s, Model={model_inference_time:.2f}s, Save={save_time:.2f}s) | Avg={avg_time:.2f}s, Freq={1 / avg_time:.3f}Hz')
        return actions_real

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

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
    policy = MotusPolicy(checkpoint_path=checkpoint_path, wan_path=wan_path, vlm_path=vlm_path, config_path=str(config_path), device=device, log_dir=usr_args.get('log_dir'), task_name=usr_args.get('task_name'))
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
    TASK_ENV.take_action(action, action_type='qpos')

def reset_model(model):
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.current_state_norm = None
    model.current_instruction = ''
    model.episode_count += 1
    model.step_count = 0
    logger.info(f'Model reset completed for episode {model.episode_count}')
