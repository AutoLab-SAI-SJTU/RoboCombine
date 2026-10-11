from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from xvla_progress_predictor._vendor import Florence2Config, Florence2ForConditionalGeneration


def maybe_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def extract_vlm_state_dict_from_xvla(model_safetensors_path: str | Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(Path(model_safetensors_path), framework="pt", device="cpu") as source:
        for key in source.keys():
            if key.startswith("vlm."):
                tensors[key.removeprefix("vlm.")] = source.get_tensor(key)
    if not tensors:
        raise ValueError(f"No vlm.* tensors found in {model_safetensors_path}")
    return tensors


def extract_vlm_state_dict_from_florence2(model_safetensors_path: str | Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    skipped_prefixes = (
        "language_model.model.decoder.",
        "language_model.lm_head.",
    )
    with safe_open(Path(model_safetensors_path), framework="pt", device="cpu") as source:
        for key in source.keys():
            if key.startswith(skipped_prefixes):
                continue
            tensors[key] = source.get_tensor(key)
    if not tensors:
        raise ValueError(f"No Florence-2 tensors found in {model_safetensors_path}")
    return tensors


class StandaloneVLMBackbone:
    def __init__(self, vlm: Florence2ForConditionalGeneration, device: torch.device) -> None:
        self.vlm = vlm.eval()
        self.device = device
        for parameter in self.vlm.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def _build_vlm_from_florence_config(cls, florence_config_payload: dict[str, Any]) -> Florence2ForConditionalGeneration:
        config = Florence2Config(**florence_config_payload)
        vlm = Florence2ForConditionalGeneration(config).to(torch.float32)
        if hasattr(vlm, "language_model"):
            language_model = vlm.language_model
            if hasattr(language_model, "model") and hasattr(language_model.model, "decoder"):
                del language_model.model.decoder
            if hasattr(language_model, "lm_head"):
                del language_model.lm_head
        return vlm

    @classmethod
    def _load_vlm_state(cls, vlm: Florence2ForConditionalGeneration, state_dict: dict[str, torch.Tensor]) -> Florence2ForConditionalGeneration:
        missing, unexpected = vlm.load_state_dict(state_dict, strict=False)
        missing_keys = set(missing)
        unexpected_keys = set(unexpected)
        allowed_missing = {"language_model.model.encoder.embed_tokens.weight"}
        if missing_keys - allowed_missing or unexpected_keys:
            raise ValueError(
                f"Standalone bundle load mismatch: missing={sorted(missing_keys)}, unexpected={sorted(unexpected_keys)}"
            )
        if missing_keys:
            vlm.language_model.model.encoder.embed_tokens.weight.data.copy_(
                vlm.language_model.model.shared.weight.data
            )
        return vlm

    @classmethod
    def from_xvla_dir(cls, model_dir: str | Path, device: torch.device) -> "StandaloneVLMBackbone":
        model_dir = Path(model_dir).resolve()
        base_config = json.loads((model_dir / "config.json").read_text())
        vlm = cls._build_vlm_from_florence_config(base_config["florence_config"])
        state_dict = extract_vlm_state_dict_from_xvla(model_dir / "model.safetensors")
        cls._load_vlm_state(vlm, state_dict)
        vlm.to(device)
        return cls(vlm=vlm, device=device)

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path, device: torch.device) -> "StandaloneVLMBackbone":
        bundle_dir = Path(bundle_dir).resolve()
        config_payload = json.loads((bundle_dir / "backbone_config.json").read_text())
        vlm = cls._build_vlm_from_florence_config(config_payload["florence_config"])
        state_dict = load_file(bundle_dir / "model.safetensors", device="cpu")
        cls._load_vlm_state(vlm, state_dict)
        vlm.to(device)
        return cls(vlm=vlm, device=device)

    @torch.no_grad()
    def extract_pooled_features(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_mask: torch.Tensor,
        mixed_precision: bool,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        pixel_values = pixel_values.to(self.device)
        image_mask = image_mask.to(self.device)
        with maybe_autocast(self.device, mixed_precision):
            encoded = self.forward_vlm(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_mask=image_mask,
            )
        return encoded["vlm_features"].mean(dim=1).to(torch.float32)

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        if int(flat_mask.sum().item()) == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_features = self.vlm._encode_image(valid_images)
        num_tokens, hidden_dim = valid_features.shape[1:]

        image_features = valid_features.new_zeros((batch_size * num_views, num_tokens, hidden_dim))
        image_features[flat_mask] = valid_features
        image_features = image_features.view(batch_size, num_views, num_tokens, hidden_dim)

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )
        encoded = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]
        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": encoded, "aux_visual_inputs": aux_visual_inputs}
