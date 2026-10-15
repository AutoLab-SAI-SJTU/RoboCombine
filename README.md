<div align="center">

<h1>
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Bar%20Chart.png" alt="Bar Chart" width="40" height="40" />
  [NeurIPS2026] Exploring the Limits of Combinatorial Generalization in
Embodied Manipulation
</h1>

<div align="center">
  <a href="https://arxiv.org/pdf/2610.15425">
    <img src="https://img.shields.io/badge/arXiv-2610.15425-b31b1b?logo=arxiv" alt="Paper PDF on arXiv">
  </a>
  <a href="https://huggingface.co/papers/2610.15425">
    <img src="https://img.shields.io/badge/HuggingFace-Paper-yellow?logo=huggingface&logoColor=yellow" alt="Paper on Hugging Face">
  </a>
  <a href="https://huggingface.co/datasets/AutoLab-SJTU/RoboCombine">
    <img src="https://img.shields.io/badge/HuggingFace-RoboCombine-yellow?logo=huggingface&logoColor=yellow">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-lightgrey">
  </a>
</div>

<p align="center">
  3 tracks: <code>full_inst</code>, <code>oracle_hse</code>, <code>ordered_hse</code><br>
  2 policy adapters: <code>Motus</code>, <code>pi05</code><br>
</p>

</div>

## 📢 News
> **[2026-10]** 🎉 Evaluation code of RoboCombine is available now.

## ✨ What does RoboCombine provide?

| Item | Public scope |
| --- | --- |
| Benchmark tasks| `19` compositional tasks |
| Supported tracks | `full_inst`, `oracle_hse`, `ordered_hse` |
| Supported policies | `Motus`, `pi05` |
| Primary metric | `sr_ordered` |

## ⚙️ Runtime setup

For evaluation, `Motus` and `pi05` must use separate Python runtimes:

Start from the detailed runtime guides:

- [docs/runtime_envs.md](docs/runtime_envs.md)
- [policy/README.md](policy/README.md)
- [policy/Motus/README.md](policy/Motus/README.md)
- [policy/pi05/README.md](policy/pi05/README.md)

Common runtime entrypoints:

```bash
export MOTUS_PYTHON_BIN=/path/to/motus/python
export PI05_PYTHON_BIN=/path/to/pi05/python3.11
export ROBOCOMBINE_ASSETS_DIR=/path/to/simulator/assets
```

If you only want a repo-local environment for source validation or packaging,
use `uv` in the repository root:

```bash
uv venv .venv --python 3.11
uv sync
```

## 🚀 Evaluation

### Quick Command Index

| If you want ... | Recommended entry |
| --- | --- |
| Inspect the released task split | `run_split.py --list-tasks` |
| Run `Motus` on `full_inst` | See `2. Motus on full_inst` below |
| Run `Motus` on `oracle_hse` | See `3. Motus on oracle_hse` below |
| Run `Motus` on `ordered_hse` | See `4. Motus on ordered_hse` below |
| Run `pi05` on `oracle_hse` | See `5. pi05 on oracle_hse` below |

<details>
<summary><strong>1. Inspect the split and list supported tasks</strong> (<code>handoff_eval/runner/run_split.py</code>)</summary>

Use this when you want to verify the public split before starting a long run.

**Relevant spec**

- `handoff_eval/specs/robocombine.json`

**Recommended command**

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

**What the released split currently declares**

- supported tracks: `full_inst`, `oracle_hse`, `ordered_hse`
- supported policies: `Motus`, `pi05`
- default episodes: `5`
- default max steps per episode: `1500`

</details>

<details>
<summary><strong>2. Motus on full_inst</strong></summary>

Use this when you want long-instruction evaluation with the `Motus` policy.

**Required external paths**

- `ROBOCOMBINE_ASSETS_DIR`
- `--checkpoint`
- `--motus-wan-dir`
- `--motus-vlm-dir`

**Recommended command**

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

**Useful follow-up commands**

```bash
tail -f handoff_eval_logs/robocombine_full_inst.out
kill "$(cat handoff_eval_logs/robocombine_full_inst.pid)"
```

</details>

<details>
<summary><strong>3. Motus on oracle_hse</strong></summary>

Use this when you want the oracle HSE track with the `Motus` policy.

**Required external paths**

- `ROBOCOMBINE_ASSETS_DIR`
- `--checkpoint`
- `--motus-wan-dir`
- `--motus-vlm-dir`

**Recommended command**

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

**Useful follow-up commands**

```bash
tail -f handoff_eval_logs/robocombine_oracle_hse.out
kill "$(cat handoff_eval_logs/robocombine_oracle_hse.pid)"
```

</details>

<details>
<summary><strong>4. Motus on ordered_hse</strong></summary>

Use this when you want predictor-guided `ordered_hse` evaluation with
`Motus`.

**Required external paths**

- `ROBOCOMBINE_ASSETS_DIR`
- `--checkpoint`
- `--motus-wan-dir`
- `--motus-vlm-dir`
- `--real-predictor-bundle-dir`
- `--real-xvla-model-dir`

**Recommended command**

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

**Useful follow-up commands**

```bash
tail -f handoff_eval_logs/robocombine_ordered_hse.out
kill "$(cat handoff_eval_logs/robocombine_ordered_hse.pid)"
```

**Note**

- In `ordered_hse`, predictor switching is recorded as a switching decision only.
- It is not written back as environment-confirmed completion.

</details>

<details>
<summary><strong>5. pi05 on oracle_hse</strong></summary>

Use this when you want to evaluate the `pi05/OpenPI` adapter on the public
`oracle_hse` track.

**Required external paths**

- `ROBOCOMBINE_ASSETS_DIR`
- `--checkpoint`
- `--pi05-train-config`
- `--pi05-model-name`
- `--pi05-checkpoint-id`
- `--pi05-step`

**Recommended command**

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

**Useful follow-up commands**

```bash
tail -f handoff_eval_logs/robocombine_pi05_oracle_hse.out
kill "$(cat handoff_eval_logs/robocombine_pi05_oracle_hse.pid)"
```

</details>

### Language switching and action chunks

Both adapters pass the active atomic instruction to their next policy inference.
Use `--clear-cache true` (default) with `run_split.py` to discard remaining actions
when a handoff cue fires, or `--clear-cache false` to latch the cue and finish the
current execution chunk before switching. The direct `script/eval_policy.py`
option is `--clear_cache true|false`; YAML accepts `clear_cache: false` as well.
For the paper's chunk-preserving setting, select `--clear-cache false` explicitly.

Oracle and predictor cues are checked after each executed action. Ordered-HSE
uses the fresh post-action observation; a deferred switch resets the predictor
only when the next instruction becomes active. For pi05, the execution chunk is
the first `--pi05-step` actions returned by the policy (as in the existing adapter).

`tests/test_language_switch_runtime.py` exercises both real adapters and the
runtime controller without GPU checkpoints. It captures the pi05 inference
`prompt` and Motus text-encoder input, checking both cache settings, oracle and
predicted handoffs, chunk boundaries, and unchanged full-instruction execution.
These contract tests do not replace checkpoint-backed simulator evaluation.

## 📦 Checkpoints and external assets

The public `RoboCombine` source release does not bundle checkpoints. Runtime
paths must always be injected explicitly through CLI flags or environment
variables.

Recommended upstream sources for `Motus`:

- `https://huggingface.co/motus-robotics/Motus_robotwin2`
- `https://huggingface.co/motus-robotics/Motus`
- `https://huggingface.co/motus-robotics/Motus_Wan2_2_5B_pretrain`
- `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct`
- `https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B`

Runtime path contract:

- `ROBOCOMBINE_ASSETS_DIR` points to the external simulator assets root.
- `--checkpoint` points to the active policy checkpoint root.
- `--motus-wan-dir` or `ROBOCOMBINE_MOTUS_WAN_DIR` points to WAN runtime assets.
- `--motus-vlm-dir` or `ROBOCOMBINE_MOTUS_VLM_DIR` points to the VLM/tokenizer directory.
- `--real-predictor-bundle-dir` and `--real-xvla-model-dir` are required for `ordered_hse`.

## 🗂️ Outputs

Each evaluation run writes to:

```text
handoff_eval_runs/robocombine/<track_tag>/<policy>/<run_id>/
```

Important artifacts:

```text
run_manifest.json
summary.json
summary.md
tasks/<task_name>/task.log
tasks/<task_name>/evaluation_statistics.json
```

`summary.json` and `summary.md` are the primary entrypoints for comparing runs.

## ✅ Verification

```bash
uv run pytest
uv build --sdist
```
