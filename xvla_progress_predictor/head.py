from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

DEFAULT_TRIGGER_THRESHOLD = 0.8


@dataclass(frozen=True)
class HeadSpec:
    input_dim: int
    hidden_dim: int
    num_layers: int
    gru_dropout: float
    mlp_hidden_dim: int
    mlp_dropout: float
    feature_pooling: str = "mean_encoder"
    trigger_threshold: float = DEFAULT_TRIGGER_THRESHOLD
    feature_mixed_precision: bool = True

    @classmethod
    def from_training_checkpoint(cls, checkpoint: dict[str, Any]) -> "HeadSpec":
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
        gru_cfg = config["model"]["gru"]
        mlp_cfg = config["model"].get("mlp", {})

        spec = cls(
            input_dim=int(state_dict["gru.weight_ih_l0"].shape[1]),
            hidden_dim=int(gru_cfg.get("hidden_dim", 256)),
            num_layers=int(gru_cfg.get("num_layers", 2)),
            gru_dropout=float(gru_cfg.get("dropout", 0.1)),
            mlp_hidden_dim=int(mlp_cfg.get("hidden_dim", 128)),
            mlp_dropout=float(mlp_cfg.get("dropout", 0.1)),
            feature_pooling=str(config["features"].get("pooling", "mean_encoder")),
            trigger_threshold=float(config["evaluation"].get("threshold", DEFAULT_TRIGGER_THRESHOLD)),
            feature_mixed_precision=bool(config["features"].get("mixed_precision", True)),
        )
        spec.validate_state_dict(state_dict)
        return spec

    def validate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        expected_shapes = {
            "gru.weight_ih_l0": (self.hidden_dim * 3, self.input_dim),
            "gru.weight_hh_l0": (self.hidden_dim * 3, self.hidden_dim),
            "head.0.weight": (self.mlp_hidden_dim, self.hidden_dim),
            "head.3.weight": (1, self.mlp_hidden_dim),
        }
        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(state_dict[key].shape)
            if actual_shape != expected_shape:
                raise ValueError(f"Unexpected shape for {key}: expected {expected_shape}, got {actual_shape}")
        for layer_idx in range(self.num_layers):
            if f"gru.weight_ih_l{layer_idx}" not in state_dict:
                raise ValueError(f"Missing GRU layer {layer_idx} weights in progress checkpoint")


class ProgressHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        gru_dropout: float,
        mlp_hidden_dim: int,
        mlp_dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid(),
        )

    @classmethod
    def from_spec(cls, spec: HeadSpec) -> "ProgressHead":
        return cls(
            input_dim=spec.input_dim,
            hidden_dim=spec.hidden_dim,
            num_layers=spec.num_layers,
            gru_dropout=spec.gru_dropout,
            mlp_hidden_dim=spec.mlp_hidden_dim,
            mlp_dropout=spec.mlp_dropout,
        )

    def forward_sequence(
        self,
        features: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, hidden_state = self.gru(features, hidden_state)
        return self.head(outputs), hidden_state

    def forward_step(
        self,
        feature: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim == 1:
            feature = feature.unsqueeze(0).unsqueeze(0)
        elif feature.ndim == 2:
            feature = feature.unsqueeze(1)
        elif feature.ndim != 3:
            raise ValueError(f"Expected 1D/2D/3D feature tensor, got shape {tuple(feature.shape)}")
        outputs, hidden_state = self.gru(feature, hidden_state)
        return self.head(outputs[:, -1]), hidden_state


def load_progress_checkpoint(progress_ckpt_path: str | Path) -> tuple[HeadSpec, dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(Path(progress_ckpt_path), map_location="cpu", weights_only=False)
    spec = HeadSpec.from_training_checkpoint(checkpoint)
    return spec, checkpoint["model_state_dict"], checkpoint
