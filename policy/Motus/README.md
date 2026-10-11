# Motus Policy

This directory contains only the `Motus` adapter code for `RoboCombine`. It
does not bundle runtime environments, model weights, WAN assets, or
VLM/tokenizer directories.

## Path Baseline

All placeholder paths below should be replaced with your own runtime
locations.

## Environment

`Motus` should not be merged into the shared `RoboCombine` root environment.
The recommended setup is to use a dedicated evaluation Python from the
runtime root that owns the `Motus` stack:

```text
/path/to/motus/python
```

`RoboCombine` only consumes that Python executable. It does not duplicate the
`Motus` runtime in-repo and does not commit any local `.venv`.

Use `Python 3.10` for the Motus runtime.

If you do not already have this environment, create a dedicated `uv` venv and
install the `Motus` dependencies:

```bash
export MOTUS_RUNTIME_ROOT=/path/to/motus/runtime/root
export MOTUS_REQUIREMENTS_FILE=/path/to/motus/requirements.txt

cd "$MOTUS_RUNTIME_ROOT"
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r "$MOTUS_REQUIREMENTS_FILE"
```

If your simulator dependencies are installed another way, add them to
the same environment. The public docs only constrain the final interpreter path,
not the low-level installation workflow on your machine.

## Required External Paths

Running `Motus` evaluation requires these external paths:

- `--checkpoint`
  - `Motus` policy checkpoint
- `--motus-wan-dir`
  - WAN runtime assets
- `--motus-vlm-dir`
  - VLM/tokenizer directory
- `ROBOCOMBINE_ASSETS_DIR`
  - simulator assets root

`ordered_hse` additionally requires:

- `--real-predictor-bundle-dir`
- `--real-xvla-model-dir`

Example placeholder values:

```bash
export MOTUS_PYTHON_BIN=/path/to/motus/python
export ROBOCOMBINE_ASSETS_DIR=/path/to/simulator/assets

export MOTUS_CHECKPOINT=/path/to/motus/checkpoint
export MOTUS_WAN_DIR=/path/to/motus/wan_assets
export MOTUS_VLM_DIR=/path/to/motus/vlm_directory

export REAL_PREDICTOR_BUNDLE_DIR=/path/to/xvla_progress_predictor_bundle
export REAL_XVLA_MODEL_DIR=/path/to/xvla_backbone
```

You can also inject the runtime directories through environment variables:

- `ROBOCOMBINE_MOTUS_WAN_DIR`
- `ROBOCOMBINE_MOTUS_VLM_DIR`

## Download URLs

Recommended upstream sources:

- Motus RoboTwin2 checkpoint
  - `https://huggingface.co/motus-robotics/Motus_robotwin2`
- Motus latent-action pretrain
  - `https://huggingface.co/motus-robotics/Motus`
- Motus WAN pretrain
  - `https://huggingface.co/motus-robotics/Motus_Wan2_2_5B_pretrain`
- Qwen3-VL-2B-Instruct
  - `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct`
- Wan2.2-TI2V-5B
  - `https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B`

## Example

```bash
export MOTUS_PYTHON_BIN=/path/to/motus/python
export ROBOCOMBINE_ASSETS_DIR=/path/to/simulator/assets

export MOTUS_CHECKPOINT=/path/to/motus/checkpoint
export MOTUS_WAN_DIR=/path/to/motus/wan_assets
export MOTUS_VLM_DIR=/path/to/motus/vlm_directory

mkdir -p handoff_eval_logs
nohup "$MOTUS_PYTHON_BIN" handoff_eval/runner/run_split.py \
  --spec robocombine \
  --track oracle_hse \
  --policy Motus \
  --checkpoint "$MOTUS_CHECKPOINT" \
  --motus-wan-dir "$MOTUS_WAN_DIR" \
  --motus-vlm-dir "$MOTUS_VLM_DIR" \
  --output-root handoff_eval_runs \
  > handoff_eval_logs/robocombine_motus.out 2>&1 &
echo $! > handoff_eval_logs/robocombine_motus.pid
```

Stop the run:

```bash
kill "$(cat handoff_eval_logs/robocombine_motus.pid)"
```

## Runtime Contract Tests

From the repository root, run `python -m pytest tests` with NumPy, PyYAML,
PyTorch, Pillow, and pytest installed. The tests exercise language handoffs and
both `clear_cache` settings using fake inference backends. They do not replace
GPU inference or simulator evaluation with model checkpoints.
