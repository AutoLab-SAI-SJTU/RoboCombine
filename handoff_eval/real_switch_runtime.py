from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
REPO_ROOT = Path(__file__).resolve().parents[1]

def _resolve_existing_path(path_text: str | None, *, base_dir: Path | None=None) -> Path | None:
    if path_text in (None, ''):
        return None
    candidate = Path(str(path_text)).expanduser()
    if not candidate.is_absolute():
        candidate = ((base_dir or Path.cwd()) / candidate).resolve()
    if candidate.exists():
        return candidate
    return None

def resolve_predictor_assets_dir(bundle_dir: str | None) -> Path:
    candidates: list[Path] = []
    explicit = _resolve_existing_path(bundle_dir)
    if explicit is not None:
        candidates.append(explicit)
    env_candidate = _resolve_existing_path(os.environ.get('ROBOTWIN_XVLA_PREDICTOR_BUNDLE_DIR', ''))
    if env_candidate is not None:
        candidates.append(env_candidate)
    for candidate in candidates:
        progress_config = candidate / 'progress_config.json'
        progress_head = candidate / 'progress_head.safetensors'
        if candidate.is_dir() and progress_config.is_file() and progress_head.is_file():
            return candidate
    raise FileNotFoundError('Authoritative XVLA real-mode predictor assets not found. Pass --real_predictor_bundle_dir or set ROBOTWIN_XVLA_PREDICTOR_BUNDLE_DIR.')

def resolve_xvla_model_dir(model_dir: str | None) -> Path | None:
    candidates: list[Path] = []
    explicit = _resolve_existing_path(model_dir)
    if explicit is not None:
        candidates.append(explicit)
    env_candidate = _resolve_existing_path(os.environ.get('ROBOTWIN_XVLA_MODEL_DIR', ''))
    if env_candidate is not None:
        candidates.append(env_candidate)
    for candidate in candidates:
        if (candidate / 'config.json').is_file() and (candidate / 'model.safetensors').is_file():
            return candidate
    return None

def load_bundle_progress_config(bundle_dir: Path) -> dict[str, Any]:
    progress_config_path = bundle_dir / 'progress_config.json'
    if not progress_config_path.is_file():
        return {}
    return json.loads(progress_config_path.read_text(encoding='utf-8'))

def ensure_xvla_predictor_import():
    try:
        from xvla_progress_predictor import XVLAProgressPredictor
        from xvla_progress_predictor.predictor import build_processor, resolve_device
        from xvla_progress_predictor.backbones import StandaloneVLMBackbone
        from xvla_progress_predictor.head import DEFAULT_TRIGGER_THRESHOLD, HeadSpec, ProgressHead
        from safetensors.torch import load_file
    except Exception as exc:
        raise ImportError('Failed to import vendored xvla_progress_predictor package from RoboCombine repository.') from exc
    return {'XVLAProgressPredictor': XVLAProgressPredictor, 'build_processor': build_processor, 'resolve_device': resolve_device, 'StandaloneVLMBackbone': StandaloneVLMBackbone, 'DEFAULT_TRIGGER_THRESHOLD': DEFAULT_TRIGGER_THRESHOLD, 'HeadSpec': HeadSpec, 'ProgressHead': ProgressHead, 'load_file': load_file}

def extract_camera_frame(observation: dict[str, Any], camera_key: str) -> np.ndarray | None:
    obs_data = observation.get('observation')
    if isinstance(obs_data, dict):
        camera_payload = obs_data.get(camera_key)
        if isinstance(camera_payload, dict) and 'rgb' in camera_payload:
            return np.asarray(camera_payload['rgb'])
    camera_payload = observation.get(camera_key)
    if isinstance(camera_payload, dict) and 'rgb' in camera_payload:
        return np.asarray(camera_payload['rgb'])
    if camera_payload is not None:
        return np.asarray(camera_payload)
    return None

def normalize_rgb_frame(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        return None
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array

def _build_predictor_from_backbone(*, assets_dir: Path, xvla_model_dir: Path, device: str):
    modules = ensure_xvla_predictor_import()
    resolved_device = modules['resolve_device'](device)
    progress_config = load_bundle_progress_config(assets_dir)
    head_cfg = progress_config['head']
    spec = modules['HeadSpec'](input_dim=int(head_cfg['input_dim']), hidden_dim=int(head_cfg['hidden_dim']), num_layers=int(head_cfg['num_layers']), gru_dropout=float(head_cfg['gru_dropout']), mlp_hidden_dim=int(head_cfg['mlp_hidden_dim']), mlp_dropout=float(head_cfg['mlp_dropout']), feature_pooling=str(progress_config.get('feature_pooling', 'mean_encoder')), trigger_threshold=float(progress_config.get('trigger_threshold', modules['DEFAULT_TRIGGER_THRESHOLD'])), feature_mixed_precision=bool(progress_config.get('feature_mixed_precision', True)))
    processor = modules['build_processor'](xvla_model_dir)
    feature_backbone = modules['StandaloneVLMBackbone'].from_xvla_dir(xvla_model_dir, resolved_device)
    progress_head = modules['ProgressHead'].from_spec(spec)
    progress_head.load_state_dict(modules['load_file'](assets_dir / 'progress_head.safetensors', device='cpu'))
    progress_head.to(resolved_device)
    progress_head.eval()
    return modules['XVLAProgressPredictor'](feature_backbone=feature_backbone, processor=processor, progress_head=progress_head, device=resolved_device, trigger_threshold=spec.trigger_threshold, feature_mixed_precision=spec.feature_mixed_precision)

class XVLARealSwitchBackend:

    def __init__(self, *, bundle_dir: str, xvla_model_dir: str | None, camera_key: str, device: str) -> None:
        self.assets_dir_path = resolve_predictor_assets_dir(bundle_dir)
        self.bundle_dir = str(self.assets_dir_path)
        self.bundle_progress_config = load_bundle_progress_config(self.assets_dir_path)
        self.bundle_camera_key = str(self.bundle_progress_config.get('camera_key') or '')
        self.camera_key = str(camera_key or self.bundle_camera_key or 'head_camera')
        self.device = str(device or 'auto')
        self.xvla_model_dir_path = resolve_xvla_model_dir(xvla_model_dir)
        self.xvla_model_dir = str(self.xvla_model_dir_path) if self.xvla_model_dir_path is not None else None
        self.uses_standalone_bundle = (self.assets_dir_path / 'model.safetensors').is_file()
        self._predictor = None

    def _load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if self.uses_standalone_bundle:
            modules = ensure_xvla_predictor_import()
            self._predictor = modules['XVLAProgressPredictor'].from_standalone(bundle_dir=self.bundle_dir, device=self.device)
            return self._predictor
        if self.xvla_model_dir_path is None:
            raise FileNotFoundError('No standalone bundle backbone found in predictor assets dir, and no X-VLA model dir was provided. Pass --real_xvla_model_dir pointing to the downloaded X-VLA-RoboTwin2 model directory.')
        self._predictor = _build_predictor_from_backbone(assets_dir=self.assets_dir_path, xvla_model_dir=self.xvla_model_dir_path, device=self.device)
        return self._predictor

    def begin_stage(self, *, instruction: str, stage_index: int, episode_step: int, can_switch: bool) -> dict[str, Any]:
        predictor = self._load_predictor()
        predictor.reset(instruction)
        return {'backend': 'xvla', 'predictor_bundle_dir': self.bundle_dir, 'xvla_model_dir': self.xvla_model_dir, 'predictor_source': 'standalone_bundle' if self.uses_standalone_bundle else 'repo_head_plus_xvla_backbone', 'camera_key': self.camera_key, 'bundle_camera_key': self.bundle_camera_key or None}

    def observe(self, observation: dict[str, Any], *, env_step: int) -> dict[str, Any]:
        frame = normalize_rgb_frame(extract_camera_frame(observation, self.camera_key))
        if frame is None:
            return {'sampled': False, 'score': None, 'predictor_step_index': None, 'backend_metadata': {'backend': 'xvla', 'backend_warning': f'missing_camera:{self.camera_key}', 'camera_key': self.camera_key, 'bundle_camera_key': self.bundle_camera_key or None, 'predictor_bundle_dir': self.bundle_dir, 'xvla_model_dir': self.xvla_model_dir, 'predictor_source': 'standalone_bundle' if self.uses_standalone_bundle else 'repo_head_plus_xvla_backbone'}}
        predictor = self._load_predictor()
        result = predictor.predict_step(frame)
        return {'sampled': True, 'score': float(result['score']), 'predictor_step_index': int(result['step_index']), 'backend_metadata': {'backend': 'xvla', 'camera_key': self.camera_key, 'bundle_camera_key': self.bundle_camera_key or None, 'predictor_bundle_dir': self.bundle_dir, 'xvla_model_dir': self.xvla_model_dir, 'predictor_source': 'standalone_bundle' if self.uses_standalone_bundle else 'repo_head_plus_xvla_backbone'}}

    def close(self) -> None:
        self._predictor = None

class XVLARealSwitchSession:

    def __init__(self, *, predictor_bundle_dir: str, xvla_model_dir: str | None=None, camera_key: str='head_camera', decision_rule: str='threshold', mode: str='live', threshold: float=0.8, consecutive_frames: int=3, min_dwell_steps: int=5, score_log_interval: int=10, device: str='auto') -> None:
        self.bundle_dir = str(resolve_predictor_assets_dir(predictor_bundle_dir))
        self.xvla_model_dir = str(resolve_xvla_model_dir(xvla_model_dir)) if resolve_xvla_model_dir(xvla_model_dir) else None
        self.camera_key = str(camera_key or 'head_camera')
        self.decision_rule = self._normalize_decision_rule(decision_rule)
        self.mode = str(mode or 'live').lower()
        if self.mode not in {'live', 'dry_run'}:
            raise ValueError(f'Unsupported real switch mode: {mode}')
        self.threshold = float(threshold)
        self.consecutive_frames = max(1, int(consecutive_frames))
        self.min_dwell_steps = max(1, int(min_dwell_steps))
        self.score_log_interval = max(1, int(score_log_interval))
        self.device = str(device or 'auto')
        self._backend = XVLARealSwitchBackend(bundle_dir=self.bundle_dir, xvla_model_dir=self.xvla_model_dir, camera_key=self.camera_key, device=self.device)
        self._active_instruction = ''
        self._active_stage_index = 0
        self._scores: list[float] = []
        self._score_trace: list[dict[str, Any]] = []
        self._trigger_events: list[dict[str, Any]] = []
        self._predictor_reset_events: list[dict[str, Any]] = []
        self._pending_trigger: dict[str, Any] | None = None
        self._dry_run_trigger: dict[str, Any] | None = None
        self._latest_metadata: dict[str, Any] = {}
        self._warnings: list[str] = []
        self._can_switch = True
        self._latest_valid_score: float | None = None

    @staticmethod
    def _normalize_decision_rule(decision_rule: str) -> str:
        resolved = str(decision_rule or 'threshold').strip().lower()
        if resolved not in {'threshold'}:
            raise ValueError(f'Unsupported real switch decision rule: {decision_rule}')
        return resolved

    @staticmethod
    def _count_consecutive_at_or_above(values: list[float], threshold: float) -> int:
        consecutive = 0
        for value in reversed(values):
            if value >= threshold:
                consecutive += 1
            else:
                break
        return consecutive

    def _decision_config_metadata(self) -> dict[str, Any]:
        return {'decision_rule': self.decision_rule}

    def close(self) -> None:
        self._backend.close()

    def begin_stage(self, *, instruction: str, stage_index: int, episode_step: int, reset_reason: str, can_switch: bool=True) -> dict[str, Any]:
        backend_metadata = self._backend.begin_stage(instruction=instruction, stage_index=stage_index, episode_step=episode_step, can_switch=can_switch)
        self._active_instruction = str(instruction)
        self._active_stage_index = int(stage_index)
        self._can_switch = bool(can_switch)
        self._scores = []
        self._pending_trigger = None
        self._dry_run_trigger = None
        self._latest_valid_score = None
        self._latest_metadata = {'switch_mode': self.mode, 'backend': 'xvla', 'latest_progress_score': None, 'latest_decision_score': None, 'progress_threshold': self.threshold, 'progress_consecutive_frames': self.consecutive_frames, 'progress_min_dwell': self.min_dwell_steps, 'progress_steps_seen': 0, 'scores_tail': [], 'should_switch': False, 'would_switch': False, 'stage_index': int(stage_index), 'camera_key': self.camera_key, 'predictor_bundle_dir': self.bundle_dir, 'xvla_model_dir': self.xvla_model_dir, 'can_switch': self._can_switch}
        self._latest_metadata.update(self._decision_config_metadata())
        event = {'event': 'predictor_reset', 'reset_reason': str(reset_reason), 'episode_step': int(episode_step), 'stage_index': int(stage_index), 'instruction': str(instruction), 'camera_key': self.camera_key, 'predictor_bundle_dir': self.bundle_dir, 'xvla_model_dir': self.xvla_model_dir, 'switch_mode': self.mode, 'backend': 'xvla', 'can_switch': self._can_switch}
        event.update(self._decision_config_metadata())
        event.update(backend_metadata)
        self._predictor_reset_events.append(event)
        return dict(event)

    def observe(self, observation: dict[str, Any], *, env_step: int) -> dict[str, Any]:
        backend_result = self._backend.observe(observation, env_step=int(env_step))
        backend_metadata = dict(backend_result.get('backend_metadata') or {})
        sampled = bool(backend_result.get('sampled', False))
        score_value = backend_result.get('score')
        score = None if score_value is None else float(score_value)
        predictor_step_index = backend_result.get('predictor_step_index')
        if predictor_step_index is not None:
            predictor_step_index = int(predictor_step_index)
        backend_warning = backend_metadata.get('backend_warning')
        if backend_warning:
            warning = str(backend_warning)
            if warning not in self._warnings:
                self._warnings.append(warning)
        consecutive_above = 0
        high_trigger_condition_met = False
        trigger_reason = None
        if sampled and score is not None:
            self._scores.append(score)
            self._latest_valid_score = score
        decision_score = self._latest_valid_score
        consecutive_above = self._count_consecutive_at_or_above(self._scores, self.threshold)
        high_trigger_condition_met = consecutive_above >= self.consecutive_frames
        criteria_met = sampled and score is not None and (len(self._scores) >= self.min_dwell_steps) and high_trigger_condition_met
        if criteria_met:
            trigger_reason = 'threshold'
        metadata = {'switch_mode': self.mode, 'backend': 'xvla', 'sampled': sampled, 'backend_warning': backend_warning, 'latest_progress_score': self._latest_valid_score, 'latest_decision_score': decision_score, 'progress_threshold': self.threshold, 'progress_consecutive_frames': self.consecutive_frames, 'progress_min_dwell': self.min_dwell_steps, 'progress_steps_seen': len(self._scores), 'scores_tail': list(self._scores[-10:]), 'should_switch': False, 'would_switch': False, 'stage_index': self._active_stage_index, 'camera_key': self.camera_key, 'predictor_bundle_dir': self.bundle_dir, 'xvla_model_dir': self.xvla_model_dir, 'can_switch': self._can_switch, 'consecutive_above_threshold': int(consecutive_above), 'high_trigger_condition_met': bool(high_trigger_condition_met), 'predictor_step_index': predictor_step_index, 'env_step': int(env_step)}
        metadata.update(self._decision_config_metadata())
        metadata.update({k: v for k, v in backend_metadata.items() if k not in metadata})
        if criteria_met and self._can_switch and (score is not None):
            trigger_payload = {**metadata, 'would_switch': True, 'trigger_score': decision_score, 'trigger_raw_score': score, 'trigger_frame': predictor_step_index, 'trigger_reason': trigger_reason, 'switch_source': 'real_switch_predictor', 'instruction': self._active_instruction}
            if self.mode == 'live':
                trigger_payload['should_switch'] = True
                if self._pending_trigger is None:
                    self._pending_trigger = dict(trigger_payload)
                    self._trigger_events.append(dict(trigger_payload))
            elif self._dry_run_trigger is None:
                self._dry_run_trigger = dict(trigger_payload)
                self._trigger_events.append(dict(trigger_payload))
            metadata.update(trigger_payload)
        self._latest_metadata = dict(metadata)
        self._score_trace.append({'env_step': int(env_step), 'predictor_step_index': predictor_step_index, 'stage_index': self._active_stage_index, 'backend': 'xvla', 'sampled': sampled, 'score': score, 'decision_score': decision_score, 'backend_warning': backend_warning, 'threshold': self.threshold, 'consecutive_above_threshold': int(consecutive_above), 'high_trigger_condition_met': bool(high_trigger_condition_met), 'trigger_reason': trigger_reason, 'would_switch': bool(metadata['would_switch']), 'should_switch': bool(metadata['should_switch']), 'switch_mode': self.mode, 'decision_rule': self.decision_rule, 'camera_key': self.camera_key, 'predictor_bundle_dir': self.bundle_dir, 'xvla_model_dir': self.xvla_model_dir})
        return dict(metadata)

    def maybe_switch(self) -> dict[str, Any]:
        if self._pending_trigger is not None:
            return dict(self._pending_trigger)
        if self._dry_run_trigger is not None:
            return dict(self._dry_run_trigger)
        return dict(self._latest_metadata)

    def export_payload(self) -> dict[str, Any]:
        return {'real_switch': {'backend': 'xvla', 'mode': self.mode, 'camera_key': self.camera_key, 'predictor_bundle_dir': self.bundle_dir, 'xvla_model_dir': self.xvla_model_dir, 'decision_rule': self.decision_rule, 'threshold': self.threshold, 'consecutive_frames': self.consecutive_frames, 'min_dwell_steps': self.min_dwell_steps, 'score_log_interval': self.score_log_interval, 'device': self.device}, 'score_trace': list(self._score_trace), 'trigger_events': list(self._trigger_events), 'predictor_reset_events': list(self._predictor_reset_events), 'warnings': list(self._warnings), 'latest_metadata': dict(self._latest_metadata)}
