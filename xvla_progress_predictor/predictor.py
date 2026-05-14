from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from transformers import AutoImageProcessor, BartTokenizer, BartTokenizerFast

from xvla_progress_predictor._vendor import XVLAProcessor
from xvla_progress_predictor.backbones import StandaloneVLMBackbone
from xvla_progress_predictor.head import DEFAULT_TRIGGER_THRESHOLD, HeadSpec, ProgressHead, load_progress_checkpoint

ImageLike = Image.Image | np.ndarray | torch.Tensor


class ProgressStepResult(TypedDict):
    score: float
    triggered: bool
    step_index: int


class XVLAProgressPredictor:
    def __init__(
        self,
        feature_backbone: StandaloneVLMBackbone,
        processor: XVLAProcessor,
        progress_head: ProgressHead,
        device: torch.device,
        trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD,
        feature_mixed_precision: bool = True,
    ) -> None:
        self.feature_backbone = feature_backbone
        self.processor = processor
        self.progress_head = progress_head.eval()
        self.device = device
        self.trigger_threshold = float(trigger_threshold)
        self.feature_mixed_precision = bool(feature_mixed_precision)
        self._cached_input_ids: torch.Tensor | None = None
        self._hidden_state: torch.Tensor | None = None
        self._step_index = 0

    @classmethod
    def from_xvla(
        cls,
        model_dir: str,
        progress_ckpt: str,
        device: str = "auto",
    ) -> "XVLAProgressPredictor":
        resolved_device = resolve_device(device)
        processor = build_processor(model_dir)
        feature_backbone = StandaloneVLMBackbone.from_xvla_dir(model_dir, resolved_device)
        spec, state_dict, _ = load_progress_checkpoint(progress_ckpt)
        progress_head = ProgressHead.from_spec(spec)
        progress_head.load_state_dict(state_dict)
        progress_head.to(resolved_device)
        progress_head.eval()
        return cls(
            feature_backbone=feature_backbone,
            processor=processor,
            progress_head=progress_head,
            device=resolved_device,
            trigger_threshold=spec.trigger_threshold,
            feature_mixed_precision=spec.feature_mixed_precision,
        )

    @classmethod
    def from_standalone(
        cls,
        bundle_dir: str,
        device: str = "auto",
    ) -> "XVLAProgressPredictor":
        resolved_device = resolve_device(device)
        bundle_dir_path = Path(bundle_dir).resolve()
        progress_config = json.loads((bundle_dir_path / "progress_config.json").read_text())
        processor = build_processor(bundle_dir_path)
        feature_backbone = StandaloneVLMBackbone.from_bundle(bundle_dir_path, resolved_device)
        head_cfg = progress_config["head"]
        spec = HeadSpec(
            input_dim=int(head_cfg["input_dim"]),
            hidden_dim=int(head_cfg["hidden_dim"]),
            num_layers=int(head_cfg["num_layers"]),
            gru_dropout=float(head_cfg["gru_dropout"]),
            mlp_hidden_dim=int(head_cfg["mlp_hidden_dim"]),
            mlp_dropout=float(head_cfg["mlp_dropout"]),
            feature_pooling=str(progress_config.get("feature_pooling", "mean_encoder")),
            trigger_threshold=float(progress_config.get("trigger_threshold", DEFAULT_TRIGGER_THRESHOLD)),
            feature_mixed_precision=bool(progress_config.get("feature_mixed_precision", True)),
        )
        progress_head = ProgressHead.from_spec(spec)
        progress_head.load_state_dict(load_file(bundle_dir_path / "progress_head.safetensors", device="cpu"))
        progress_head.to(resolved_device)
        progress_head.eval()
        return cls(
            feature_backbone=feature_backbone,
            processor=processor,
            progress_head=progress_head,
            device=resolved_device,
            trigger_threshold=spec.trigger_threshold,
            feature_mixed_precision=spec.feature_mixed_precision,
        )

    def reset(self, instruction: str) -> None:
        if not instruction.strip():
            raise ValueError("instruction must be non-empty")
        encoded = self.processor.encode_language(instruction)
        self._cached_input_ids = encoded["input_ids"].to(self.device)
        self._hidden_state = None
        self._step_index = 0

    @torch.no_grad()
    def predict_step(self, frame: ImageLike) -> ProgressStepResult:
        if self._cached_input_ids is None:
            raise RuntimeError("Call reset(instruction) before predict_step(frame)")
        pixel_values, image_mask = prepare_single_frame_inputs(self.processor, frame)
        pooled_features = self.feature_backbone.extract_pooled_features(
            input_ids=self._cached_input_ids,
            pixel_values=pixel_values,
            image_mask=image_mask,
            mixed_precision=self.feature_mixed_precision,
        )
        score_tensor, self._hidden_state = self.progress_head.forward_step(
            pooled_features,
            hidden_state=self._hidden_state,
        )
        score = float(score_tensor.squeeze().item())
        result: ProgressStepResult = {
            "score": score,
            "triggered": bool(score >= self.trigger_threshold),
            "step_index": self._step_index,
        }
        self._step_index += 1
        return result


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_processor(model_dir: str | Path) -> XVLAProcessor:
    image_processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
    try:
        tokenizer = BartTokenizerFast.from_pretrained(model_dir)
    except Exception:
        tokenizer = BartTokenizer.from_pretrained(model_dir)
    return XVLAProcessor(image_processor=image_processor, tokenizer=tokenizer)


def prepare_single_frame_inputs(processor: XVLAProcessor, frame: ImageLike) -> tuple[torch.Tensor, torch.Tensor]:
    image = to_pil_rgb_image(frame)
    pixel_values = processor.image_processor([image], return_tensors="pt")["pixel_values"].unsqueeze(1)
    image_mask = torch.ones((1, 1), dtype=torch.bool)
    return pixel_values, image_mask


def to_pil_rgb_image(frame: ImageLike) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Unsupported frame type: {type(frame)!r}")

    array = frame
    if array.ndim != 3:
        raise ValueError(f"Expected 3D frame array, got shape {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(array.max()) <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0)
    else:
        array = np.clip(array, 0, 255)
    return Image.fromarray(array.astype(np.uint8), mode="RGB")
