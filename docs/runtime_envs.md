RoboCombine Runtime Environments
================================

`RoboCombine` includes only evaluation code, specs, runners, summaries, and
policy adapters. It does not ship simulator assets, model checkpoints, Motus
WAN assets, or VLM/tokenizer directories.

The repository does not require `Motus` and `pi05` to share one Python runtime.
The recommended public contract is one dedicated Python executable per policy.

Recommended Workspace Layout
----------------------------

The public docs use placeholder paths instead of machine-specific layouts.
Replace each `/path/to/...` value with your actual runtime location:

- `Motus` Python: `/path/to/motus/python`
- `pi05` Python: `/path/to/pi05/python3.11`
- simulator assets: `/path/to/simulator/assets`

Do not copy machine-specific local paths into the public docs.

Recommended Runtime Contract
----------------------------

- `Motus`
  - reuse the dedicated evaluation environment on the `Motus` side
  - validated interpreter version: `/path/to/motus/python` (`Python 3.10`)
- `pi05`
  - reuse the dedicated `uv` environment under your `pi05` runtime root
  - validated interpreter version: `/path/to/pi05/python3.11` (`Python 3.11`)
  - `pyproject.toml` / `uv.lock` live under `/path/to/pi05/runtime/root`

pi05 Environment
----------------

`pi05` should reuse the validated `uv` environment under
the runtime root that owns the `OpenPI/pi05` stack instead of creating a second
environment under the `RoboCombine` root:

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

Why this is necessary:

- the selected `pi05` runtime root owns the `OpenPI/pi05` dependency stack
- the `RoboCombine` evaluation process must also import `curobo` through
  `envs/robot/planner.py`

Motus Environment
-----------------

`Motus` should reuse the validated evaluation environment that owns the
`Motus` runtime stack:

```text
/path/to/motus/python
```

That validated interpreter is `Python 3.10`.

If you need to build the environment from scratch, create a dedicated `uv`
environment and install the `Motus` requirements:

```bash
export MOTUS_RUNTIME_ROOT=/path/to/motus/runtime/root
export MOTUS_REQUIREMENTS_FILE=/path/to/motus/requirements.txt

cd "$MOTUS_RUNTIME_ROOT"
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r "$MOTUS_REQUIREMENTS_FILE"
```

`RoboCombine` itself only consumes the final interpreter path. It does not
duplicate the `Motus` runtime in-repo.

Run Entry
---------

Export the two interpreters:

```bash
export MOTUS_PYTHON_BIN=/path/to/motus/python
export PI05_PYTHON_BIN=/path/to/pi05/python3.11
```

`Motus` example:

```bash
"$MOTUS_PYTHON_BIN" handoff_eval/runner/run_split.py \
  --spec robocombine \
  --track oracle_hse \
  --policy Motus \
  --checkpoint /path/to/motus/checkpoint \
  --motus-wan-dir /path/to/motus/wan_assets \
  --motus-vlm-dir /path/to/motus/vlm_directory
```

`pi05` example:

```bash
"$PI05_PYTHON_BIN" handoff_eval/runner/run_split.py \
  --spec robocombine \
  --track oracle_hse \
  --policy pi05 \
  --checkpoint /path/to/pi05/checkpoint \
  --pi05-train-config pi05_aloha_full_base \
  --pi05-model-name custom_checkpoint \
  --pi05-checkpoint-id 0 \
  --pi05-step 50
```

Checkpoint Policy
-----------------

The public `RoboCombine` release does not bundle checkpoints.

The documentation keeps only upstream URLs / download instructions:

- Motus
  - `motus-robotics/Motus_robotwin2`
  - `motus-robotics/Motus`
  - `motus-robotics/Motus_Wan2_2_5B_pretrain`
- Qwen3-VL
  - `Qwen/Qwen3-VL-2B-Instruct`
- Wan2.2
  - `Wan-AI/Wan2.2-TI2V-5B`
- pi0.5 / OpenPI
  - user-supplied checkpoint paths only

Runtime path contract:

- `ROBOCOMBINE_ASSETS_DIR`
  - external simulator assets root
- `--motus-wan-dir` or `ROBOCOMBINE_MOTUS_WAN_DIR`
  - external WAN runtime assets
- `--motus-vlm-dir` or `ROBOCOMBINE_MOTUS_VLM_DIR`
  - external VLM/tokenizer directory

Scope
-----

"Self-contained" in `RoboCombine` means:

- evaluation logic is self-contained
- HSE tracks are self-contained
- split runner and summaries are self-contained
- there is no dependency on external `compare/legacy` repositories

It does not mean:

- bundling model checkpoints
- bundling simulator assets
- forcing both policies into one shared runtime
