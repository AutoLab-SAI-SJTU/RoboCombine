RoboCombine
===========

`RoboCombine` is a code-only open-source evaluation repository for the
RoboCombine compositional robot-manipulation benchmark.

This repository publishes only evaluation code, task specifications, runners,
summaries, and policy adapters.

The current public scope is intentionally fixed:

- 3 HSE variants:
  - `full_inst`
  - `oracle_hse`
  - `ordered_hse`
- 2 policies:
  - `Motus`
  - `pi05`

Legacy support outside this scope has been removed:

- `oracle_hse_reset_origin`
- `legacy_v12`
- `robo_dopamine`
- `xvla/holobrain adapter`
- external `compare/legacy` evaluation dependencies

Self-Contained Scope
--------------------

The repository is self-contained for evaluation logic:

- `handoff_eval/`: split runner, summaries, HSE controller, and Ordered HSE runtime
- `policy/Motus/`
- `policy/pi05/`
- vendored `xvla_progress_predictor/`
- vendored `openpi/`

Users still need to provide the following external runtime assets:

- simulator assets
- policy checkpoints
- predictor bundle and X-VLA backbone weights for `ordered_hse`
- WAN, VLM, tokenizer, and policy weights required by `Motus`

Repository constraints:

- no tracked symlinks
- no tracked model directories
- only upstream URLs and external path contracts are documented

Important notes:

- `Motus` and `pi05` must use separate Python runtimes
- the currently validated interpreter versions are:
  - `Motus` -> `/path/to/motus/python` (`Python 3.10`)
  - `pi05` -> `/path/to/pi05/python3.11` (`Python 3.11`)
- runtime details are documented in [docs/runtime_envs.md](docs/runtime_envs.md)
- policy-specific details are documented in [policy/README.md](policy/README.md)

Environment
-----------

Replace the placeholder paths below with your own runtime locations:

```bash
export MOTUS_PYTHON_BIN=/path/to/motus/python
export PI05_PYTHON_BIN=/path/to/pi05/python3.11
```

Runtime entrypoints:

- `Motus` runs with `"$MOTUS_PYTHON_BIN"`
- `pi05` runs with `"$PI05_PYTHON_BIN"`

RoboCombine Split
-----------------

The RoboCombine split specification lives at:

```text
handoff_eval/specs/robocombine.json
```

The spec currently declares only:

- `supported_tracks = ["full_inst", "oracle_hse", "ordered_hse"]`
- `supported_policies = ["Motus", "pi05"]`

Run Split Evaluation
--------------------

List tasks:

```bash
"$MOTUS_PYTHON_BIN" handoff_eval/runner/run_split.py \
  --spec robocombine \
  --track oracle_hse \
  --policy Motus \
  --checkpoint /path/to/checkpoint \
  --motus-wan-dir /path/to/motus/wan_assets \
  --motus-vlm-dir /path/to/Qwen3-VL-2B-Instruct \
  --list-tasks
```

`full_inst`:

```bash
mkdir -p handoff_eval_logs
nohup "$MOTUS_PYTHON_BIN" handoff_eval/runner/run_split.py \
  --spec robocombine \
  --track full_inst \
  --policy Motus \
  --checkpoint /path/to/checkpoint \
  --motus-wan-dir /path/to/motus/wan_assets \
  --motus-vlm-dir /path/to/Qwen3-VL-2B-Instruct \
  --checkpoint-label model_under_review \
  --output-root handoff_eval_runs \
  > handoff_eval_logs/robocombine_full_inst.out 2>&1 &
echo $! > handoff_eval_logs/robocombine_full_inst.pid
```

`oracle_hse`:

```bash
mkdir -p handoff_eval_logs
nohup "$MOTUS_PYTHON_BIN" handoff_eval/runner/run_split.py \
  --spec robocombine \
  --track oracle_hse \
  --policy Motus \
  --checkpoint /path/to/checkpoint \
  --motus-wan-dir /path/to/motus/wan_assets \
  --motus-vlm-dir /path/to/Qwen3-VL-2B-Instruct \
  --checkpoint-label model_under_review \
  --output-root handoff_eval_runs \
  > handoff_eval_logs/robocombine_oracle_hse.out 2>&1 &
echo $! > handoff_eval_logs/robocombine_oracle_hse.pid
```

`ordered_hse`:

```bash
mkdir -p handoff_eval_logs
nohup "$MOTUS_PYTHON_BIN" handoff_eval/runner/run_split.py \
  --spec robocombine \
  --track ordered_hse \
  --policy Motus \
  --checkpoint /path/to/checkpoint \
  --motus-wan-dir /path/to/motus/wan_assets \
  --motus-vlm-dir /path/to/Qwen3-VL-2B-Instruct \
  --checkpoint-label model_under_review \
  --real-predictor-bundle-dir /path/to/xvla_progress_predictor_bundle \
  --real-xvla-model-dir /path/to/X-VLA-RoboTwin2 \
  --output-root handoff_eval_runs \
  > handoff_eval_logs/robocombine_ordered_hse.out 2>&1 &
echo $! > handoff_eval_logs/robocombine_ordered_hse.pid
```

Stop a run:

```bash
kill "$(cat handoff_eval_logs/robocombine_ordered_hse.pid)"
```

Checkpoint Sources
------------------

The `RoboCombine` source release does not bundle checkpoints. Runtime paths must
always be supplied explicitly.

Recommended upstream sources:

- Motus RoboTwin2:
  - `https://huggingface.co/motus-robotics/Motus_robotwin2`
- Motus latent-action pretrain:
  - `https://huggingface.co/motus-robotics/Motus`
- Motus WAN pretrain:
  - `https://huggingface.co/motus-robotics/Motus_Wan2_2_5B_pretrain`
- Qwen3-VL-2B-Instruct:
  - `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct`
- Wan2.2-TI2V-5B:
  - `https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B`

Runtime path contract:

- `--checkpoint` points to the `Motus` policy checkpoint
- `--motus-wan-dir` points to external WAN runtime assets
- `--motus-vlm-dir` points to the external VLM/tokenizer directory

pi05 Example
------------

```bash
mkdir -p handoff_eval_logs
nohup "$PI05_PYTHON_BIN" handoff_eval/runner/run_split.py \
  --spec robocombine \
  --track oracle_hse \
  --policy pi05 \
  --checkpoint /path/to/pi05_checkpoint \
  --pi05-train-config pi05_aloha_full_base \
  --pi05-model-name custom_checkpoint \
  --pi05-checkpoint-id 0 \
  --pi05-step 50 \
  --output-root handoff_eval_runs \
  > handoff_eval_logs/robocombine_pi05_oracle_hse.out 2>&1 &
echo $! > handoff_eval_logs/robocombine_pi05_oracle_hse.pid
```

Outputs
-------

Each split run writes to:

```text
handoff_eval_runs/robocombine/<track_tag>/<policy>/<run_id>/
```

Important files:

```text
run_manifest.json
summary.json
summary.md
tasks/<task_name>/task.log
tasks/<task_name>/evaluation_statistics.json
```

The primary metric is `sr_ordered`.

In `ordered_hse`, predictor switching is recorded only as a switching decision.
It is not written back as environment-confirmed completion.

Verification
------------

```bash
uv run pytest
uv build --sdist
```
