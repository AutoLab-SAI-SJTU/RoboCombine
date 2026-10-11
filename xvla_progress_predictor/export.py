from __future__ import annotations

import json
import shutil
from pathlib import Path

from safetensors.torch import save_file

from xvla_progress_predictor.backbones import (
    extract_vlm_state_dict_from_florence2,
    extract_vlm_state_dict_from_xvla,
)
from xvla_progress_predictor.head import load_progress_checkpoint

BUNDLE_FORMAT_VERSION = 1
PROCESSOR_ASSET_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)


def export_standalone_bundle(
    model_dir: str | Path,
    progress_ckpt: str | Path,
    bundle_dir: str | Path,
) -> Path:
    model_dir = Path(model_dir).resolve()
    progress_ckpt = Path(progress_ckpt).resolve()
    bundle_dir = Path(bundle_dir).resolve()

    bundle_dir.mkdir(parents=True, exist_ok=True)
    if any(bundle_dir.iterdir()):
        raise FileExistsError(f"Bundle directory is not empty: {bundle_dir}")

    base_config = json.loads((model_dir / "config.json").read_text())
    florence_config = base_config["florence_config"]
    spec, state_dict, checkpoint = load_progress_checkpoint(progress_ckpt)

    save_file(extract_vlm_state_dict_from_xvla(model_dir / "model.safetensors"), bundle_dir / "model.safetensors")
    save_file(state_dict, bundle_dir / "progress_head.safetensors")

    for asset_name in PROCESSOR_ASSET_FILES:
        source_path = model_dir / asset_name
        if source_path.exists():
            shutil.copy2(source_path, bundle_dir / asset_name)

    (bundle_dir / "backbone_config.json").write_text(
        json.dumps(
            {
                "bundle_format_version": BUNDLE_FORMAT_VERSION,
                "florence_config": florence_config,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    (bundle_dir / "progress_config.json").write_text(
        json.dumps(
            {
                "bundle_format_version": BUNDLE_FORMAT_VERSION,
                "feature_pooling": spec.feature_pooling,
                "feature_mixed_precision": spec.feature_mixed_precision,
                "trigger_threshold": spec.trigger_threshold,
                "camera_key": checkpoint["config"]["data"].get("camera_key", "front_camera"),
                "head": {
                    "input_dim": spec.input_dim,
                    "hidden_dim": spec.hidden_dim,
                    "num_layers": spec.num_layers,
                    "gru_dropout": spec.gru_dropout,
                    "mlp_hidden_dim": spec.mlp_hidden_dim,
                    "mlp_dropout": spec.mlp_dropout,
                },
                "source": {
                    "model_dir": str(model_dir),
                    "progress_checkpoint": str(progress_ckpt),
                    "feature_model_path": checkpoint["config"]["features"]["model_path"],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return bundle_dir


def export_standalone_bundle_from_florence2(
    model_dir: str | Path,
    progress_ckpt: str | Path,
    bundle_dir: str | Path,
) -> Path:
    model_dir = Path(model_dir).resolve()
    progress_ckpt = Path(progress_ckpt).resolve()
    bundle_dir = Path(bundle_dir).resolve()

    bundle_dir.mkdir(parents=True, exist_ok=True)
    if any(bundle_dir.iterdir()):
        raise FileExistsError(f"Bundle directory is not empty: {bundle_dir}")

    florence_config = json.loads((model_dir / "config.json").read_text())
    spec, state_dict, checkpoint = load_progress_checkpoint(progress_ckpt)

    save_file(
        extract_vlm_state_dict_from_florence2(model_dir / "model.safetensors"),
        bundle_dir / "model.safetensors",
    )
    save_file(state_dict, bundle_dir / "progress_head.safetensors")

    for asset_name in PROCESSOR_ASSET_FILES:
        source_path = model_dir / asset_name
        if source_path.exists():
            shutil.copy2(source_path, bundle_dir / asset_name)

    (bundle_dir / "backbone_config.json").write_text(
        json.dumps(
            {
                "bundle_format_version": BUNDLE_FORMAT_VERSION,
                "florence_config": florence_config,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    (bundle_dir / "progress_config.json").write_text(
        json.dumps(
            {
                "bundle_format_version": BUNDLE_FORMAT_VERSION,
                "feature_pooling": spec.feature_pooling,
                "feature_mixed_precision": spec.feature_mixed_precision,
                "trigger_threshold": spec.trigger_threshold,
                "camera_key": checkpoint["config"]["data"].get("camera_key", "front_camera"),
                "head": {
                    "input_dim": spec.input_dim,
                    "hidden_dim": spec.hidden_dim,
                    "num_layers": spec.num_layers,
                    "gru_dropout": spec.gru_dropout,
                    "mlp_hidden_dim": spec.mlp_hidden_dim,
                    "mlp_dropout": spec.mlp_dropout,
                },
                "source": {
                    "model_dir": str(model_dir),
                    "progress_checkpoint": str(progress_ckpt),
                    "feature_model_path": checkpoint["config"]["features"]["model_path"],
                    "export_source": "florence2",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return bundle_dir
