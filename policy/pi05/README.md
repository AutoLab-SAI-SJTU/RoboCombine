# pi05 Policy

This directory contains only the `pi05/OpenPI` adapter code for `RoboCombine`.
It does not bundle checkpoints.

## Path Baseline

All placeholder paths below should be replaced with your own runtime
locations.

## Environment

`pi05` should not share the same environment as `Motus`. The recommended setup
is to reuse the dedicated `uv` environment under your `pi05` runtime root:

```bash
export PI05_RUNTIME_ROOT=/path/to/pi05/runtime/root
export CUROBO_EDITABLE_ROOT=/path/to/curobo

(
  cd "$PI05_RUNTIME_ROOT"
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  uv pip install --python .venv/bin/python3.11 -e "$CUROBO_EDITABLE_ROOT" --no-build-isolation
  uv pip install --python .venv/bin/python3.11 timm
)

export PI05_PYTHON_BIN="$PI05_RUNTIME_ROOT/.venv/bin/python3.11"
```

This matters because a single `RoboCombine` evaluation process loads both:

- `pi05/OpenPI` policy dependencies
- the `curobo` dependency chain used by `envs/robot/planner.py`

`RoboCombine` uses this interpreter directly and does not duplicate `.venv`
inside the repository.

## Required External Paths

Running `pi05` evaluation requires these external paths:

- `--checkpoint`
  - external `pi05` checkpoint root
- `ROBOCOMBINE_ASSETS_DIR`
  - simulator assets root

`ordered_hse` additionally requires:

- `--real-predictor-bundle-dir`
- `--real-xvla-model-dir`

Example placeholder values:

```bash
export PI05_PYTHON_BIN=/path/to/pi05/python3.11
export ROBOCOMBINE_ASSETS_DIR=/path/to/simulator/assets

export PI05_CHECKPOINT=/path/to/pi05/checkpoint

export REAL_PREDICTOR_BUNDLE_DIR=/path/to/xvla_progress_predictor_bundle
export REAL_XVLA_MODEL_DIR=/path/to/xvla_backbone
```

## Model Path Contract

The directory passed to `--checkpoint` must satisfy the current expectations of
`pi_model.py`:

- the directory contains `assets/`
- `assets/` contains at least one `asset_id` subdirectory

In practice, `run_split.py` turns:

- `--checkpoint /path/to/pi05/checkpoint`

into the adapter-level setting:

- `custom_ckpt_path=/path/to/pi05/checkpoint`

`RoboCombine` no longer supports loading models from an in-repo
`policy/pi05/checkpoints/...` path.

## Example

```bash
export PI05_PYTHON_BIN=/path/to/pi05/python3.11
export ROBOCOMBINE_ASSETS_DIR=/path/to/simulator/assets
export PI05_CHECKPOINT=/path/to/pi05/checkpoint

mkdir -p handoff_eval_logs
nohup "$PI05_PYTHON_BIN" handoff_eval/runner/run_split.py \
  --spec robocombine \
  --track oracle_hse \
  --policy pi05 \
  --checkpoint "$PI05_CHECKPOINT" \
  --pi05-train-config pi05_aloha_full_base \
  --pi05-model-name custom_checkpoint \
  --pi05-checkpoint-id 0 \
  --pi05-step 50 \
  --output-root handoff_eval_runs \
  > handoff_eval_logs/robocombine_pi05.out 2>&1 &
echo $! > handoff_eval_logs/robocombine_pi05.pid
```

Stop the run:

```bash
kill "$(cat handoff_eval_logs/robocombine_pi05.pid)"
```

## Runtime Contract Tests

From the repository root, run `python -m pytest tests` with NumPy, PyYAML,
PyTorch, Pillow, and pytest installed. The tests exercise language handoffs and
both `clear_cache` settings using fake inference backends. They do not replace
GPU inference or simulator evaluation with model checkpoints.
