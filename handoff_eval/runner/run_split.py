from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS_ROOT = REPO_ROOT / 'handoff_eval' / 'specs'

@dataclass
class TaskRunRecord:
    task_name: str
    stage_count: int
    result_dir: str
    log_path: str
    command_path: str
    command: list[str]
    exit_code: int
    stats_path: str | None
    seed_receipt: str | None
    used_seed_receipt: str | None

def load_spec(spec_name: str) -> dict[str, Any]:
    spec_path = SPECS_ROOT / f'{spec_name}.json'
    if not spec_path.is_file():
        raise FileNotFoundError(f'Spec not found: {spec_path}')
    with spec_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a RoboCombine split evaluation.')
    parser.add_argument('--spec', required=True, help='Spec name under handoff_eval/specs, without .json suffix')
    parser.add_argument('--track', required=True, choices=['full_inst', 'oracle_hse', 'ordered_hse'])
    parser.add_argument('--policy', required=True, choices=['Motus', 'pi05'])
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path for the selected policy')
    parser.add_argument('--checkpoint-label', default=None, help='Optional short label used in output metadata')
    parser.add_argument('--episodes', type=int, default=None, help='Override episode count; default comes from spec')
    parser.add_argument('--max-steps-per-episode', type=int, default=None, help='Override native TASK_ENV.step_lim; default can come from spec')
    parser.add_argument('--task-config', default=None, help='Override RoboTwin task_config; default comes from spec')
    parser.add_argument('--output-root', default='handoff_eval_runs', help='Root directory for split runs')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed used only when no priority seed receipt exists')
    parser.add_argument('--save-video', action='store_true', help='Enable native eval video logging')
    parser.add_argument('--task-filter', action='append', default=[], help='Run only the named task(s); repeatable')
    parser.add_argument('--max-tasks', type=int, default=None, help='Run only the first N tasks after filtering')
    parser.add_argument('--list-tasks', action='store_true', help='Print task names and exit')
    parser.add_argument('--pi05-train-config', default='pi05_aloha_full_base')
    parser.add_argument('--pi05-model-name', default='custom_checkpoint')
    parser.add_argument('--pi05-checkpoint-id', type=int, default=0)
    parser.add_argument('--pi05-step', type=int, default=50)
    parser.add_argument('--clear-cache', choices=['true', 'false'], default='true', help='Discard remaining chunk actions on handoff; false drains the chunk first')
    parser.add_argument('--motus-wan-dir', default=None, help='Path to external WAN runtime assets required by Motus')
    parser.add_argument('--motus-vlm-dir', default=None, help='Path to external VLM/tokenizer directory required by Motus')
    parser.add_argument('--ordered-hse-profile', dest='ordered_hse_profile', choices=['threshold'], default='threshold')
    parser.add_argument('--real-predictor-bundle-dir', default=None, help='Path to standalone XVLA predictor bundle')
    parser.add_argument('--real-xvla-model-dir', default=None, help='Path to downloaded X-VLA-RoboTwin2 backbone directory')
    parser.add_argument('--real-camera-key', default='head_camera')
    parser.add_argument('--real-device', default='auto')
    parser.add_argument('--real-switch-mode', choices=['live', 'dry_run'], default='live')
    parser.add_argument('--real-threshold', type=float, default=0.8, help='Threshold rule: trigger threshold')
    parser.add_argument('--real-consecutive-frames', type=int, default=3, help='Threshold rule: consecutive-frame trigger count')
    parser.add_argument('--real-min-dwell-steps', type=int, default=5)
    parser.add_argument('--real-score-log-interval', type=int, default=10)
    return parser.parse_args()

def resolve_output_root(output_root: str) -> Path:
    candidate = Path(output_root).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate

def resolve_checkpoint(path_text: str) -> Path:
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate

def derive_checkpoint_label(checkpoint_path: Path, override: str | None) -> str:
    if override:
        return override
    return checkpoint_path.name or 'checkpoint'

def filter_tasks(tasks: list[dict[str, Any]], filters: list[str], max_tasks: int | None) -> list[dict[str, Any]]:
    selected = tasks
    if filters:
        filter_set = set(filters)
        selected = [task for task in selected if task['task_name'] in filter_set]
    if max_tasks is not None:
        selected = selected[:max_tasks]
    return selected

def format_command(command: list[str]) -> str:
    return ' '.join((shlex.quote(part) for part in command))

def resolve_ordered_hse_tag(args: argparse.Namespace) -> str:
    if args.track != 'ordered_hse':
        return args.track
    return f'ordered_hse_{args.ordered_hse_profile}'

def resolve_max_steps_per_episode(args: argparse.Namespace, spec: dict[str, Any]) -> int | None:
    value = args.max_steps_per_episode
    if value is None:
        value = spec.get('default_max_steps_per_episode')
    return int(value) if value is not None else None

def read_seed_receipt(spec: dict[str, Any], task: dict[str, Any]) -> Path | None:
    relative = task.get('seed_receipt')
    if not relative:
        return None
    candidate = (SPECS_ROOT / relative).resolve()
    if candidate.is_file() and candidate.read_text(encoding='utf-8').strip():
        return candidate
    return None

def write_used_seed_receipt(task_dir: Path, task_name: str, fallback_receipt: Path | None, used_dir: Path) -> Path | None:
    used_dir.mkdir(parents=True, exist_ok=True)
    target = used_dir / f'{task_name}.txt'
    if fallback_receipt and fallback_receipt.is_file():
        target.write_text(fallback_receipt.read_text(encoding='utf-8'), encoding='utf-8')
        return target
    expert_demo_json = task_dir / f'{task_name}_expert_demo_seeds.json'
    successful_json = task_dir / f'{task_name}_successful_seeds.json'
    for candidate in [expert_demo_json, successful_json]:
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding='utf-8'))
            for key in ['expert_demo_seeds', 'successful_seeds']:
                if key in payload and payload[key]:
                    target.write_text(' '.join((str(item) for item in payload[key])) + '\n', encoding='utf-8')
                    return target
    return None

def build_eval_command(*, args: argparse.Namespace, spec: dict[str, Any], task: dict[str, Any], checkpoint_path: Path, checkpoint_label: str, task_dir: Path, seed_receipt: Path | None) -> list[str]:
    policy = args.policy
    if policy == 'Motus':
        config_path = REPO_ROOT / 'policy' / 'Motus' / 'deploy_policy.yml'
        policy_name = 'Motus'
    else:
        config_path = REPO_ROOT / 'policy' / 'pi05' / 'deploy_policy.yml'
        policy_name = 'pi05'
    command = [sys.executable, 'script/eval_policy.py', '--config', str(config_path)]
    task_config = args.task_config or spec.get('default_task_config', 'robocombine_eval')
    episodes = int(args.episodes or spec.get('default_episodes', 5))
    overrides: list[str] = ['--task_name', task['task_name'], '--task_config', task_config, '--policy_name', policy_name, '--seed', str(args.base_seed), '--episode_num', str(episodes), '--ckpt_setting', str(checkpoint_path), '--ckpt_setting_label', checkpoint_label, '--save_dir', str(task_dir), '--log_dir', str(task_dir), '--hse_variant', str(args.track)]
    overrides.extend(['--clear_cache', str(getattr(args, 'clear_cache', 'true'))])
    if args.save_video:
        overrides.extend(['--eval_video_log', 'True'])
    max_steps = resolve_max_steps_per_episode(args, spec)
    if max_steps is not None:
        overrides.extend(['--max_steps_per_episode', str(max_steps)])
    if seed_receipt is not None:
        overrides.extend(['--priority_seeds_file', str(seed_receipt)])
    if args.track == 'full_inst':
        overrides.extend(['--custom_instruction', task['instruction']['long']])
    elif args.track == 'ordered_hse':
        overrides.extend(['--real_decision_rule', args.ordered_hse_profile, '--real_camera_key', str(args.real_camera_key), '--real_device', str(args.real_device), '--real_switch_mode', str(args.real_switch_mode), '--real_min_dwell_steps', str(args.real_min_dwell_steps), '--real_score_log_interval', str(args.real_score_log_interval)])
        if args.real_predictor_bundle_dir:
            overrides.extend(['--real_predictor_bundle_dir', str(resolve_checkpoint(args.real_predictor_bundle_dir))])
        if args.real_xvla_model_dir:
            overrides.extend(['--real_xvla_model_dir', str(resolve_checkpoint(args.real_xvla_model_dir))])
        overrides.extend(['--real_threshold', str(args.real_threshold), '--real_consecutive_frames', str(args.real_consecutive_frames)])
    if policy == 'Motus':
        wan_dir = getattr(args, 'motus_wan_dir', None) or os.environ.get('ROBOCOMBINE_MOTUS_WAN_DIR')
        vlm_dir = getattr(args, 'motus_vlm_dir', None) or os.environ.get('ROBOCOMBINE_MOTUS_VLM_DIR')
        if not wan_dir:
            raise ValueError('Motus evaluation requires --motus-wan-dir or ROBOCOMBINE_MOTUS_WAN_DIR')
        if not vlm_dir:
            raise ValueError('Motus evaluation requires --motus-vlm-dir or ROBOCOMBINE_MOTUS_VLM_DIR')
        overrides.extend(['--wan_path', str(resolve_checkpoint(wan_dir)), '--vlm_path', str(resolve_checkpoint(vlm_dir))])
    else:
        overrides.extend(['--train_config_name', args.pi05_train_config, '--model_name', args.pi05_model_name, '--checkpoint_id', str(args.pi05_checkpoint_id), '--pi0_step', str(args.pi05_step), '--custom_ckpt_path', str(checkpoint_path)])
    command.extend(['--overrides', *overrides])
    return command

def run_single_task(*, args: argparse.Namespace, spec: dict[str, Any], task: dict[str, Any], checkpoint_path: Path, checkpoint_label: str, run_root: Path, used_seed_root: Path) -> TaskRunRecord:
    task_dir = run_root / 'tasks' / task['task_name']
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_dir / 'task.log'
    command_path = task_dir / 'command.sh'
    seed_receipt = read_seed_receipt(spec, task)
    command = build_eval_command(args=args, spec=spec, task=task, checkpoint_path=checkpoint_path, checkpoint_label=checkpoint_label, task_dir=task_dir, seed_receipt=seed_receipt)
    command_path.write_text(format_command(command) + '\n', encoding='utf-8')
    print(f"[run_split] START task={task['task_name']} track={args.track} policy={args.policy}")
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write(f'# cwd: {REPO_ROOT}\n')
        handle.write(f'# command: {format_command(command)}\n\n')
        handle.flush()
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    stats_path = task_dir / 'evaluation_statistics.json'
    used_seed = write_used_seed_receipt(task_dir, task['task_name'], seed_receipt, used_seed_root)
    print(f"[run_split] END task={task['task_name']} code={completed.returncode} stats={('yes' if stats_path.is_file() else 'no')}")
    return TaskRunRecord(task_name=task['task_name'], stage_count=int(task['stage_count']), result_dir=str(task_dir), log_path=str(log_path), command_path=str(command_path), command=command, exit_code=int(completed.returncode), stats_path=str(stats_path) if stats_path.is_file() else None, seed_receipt=str(seed_receipt) if seed_receipt else None, used_seed_receipt=str(used_seed) if used_seed else None)

def main() -> int:
    args = parse_args()
    spec = load_spec(args.spec)
    tasks = filter_tasks(spec['tasks'], args.task_filter, args.max_tasks)
    if args.list_tasks:
        for task in tasks:
            print(task['task_name'])
        return 0
    if not tasks:
        raise SystemExit('No tasks selected after applying filters')
    if args.track not in set(spec.get('supported_tracks', [])):
        raise SystemExit(f'Track {args.track} not supported by spec {args.spec}')
    if args.policy not in set(spec.get('supported_policies', [])):
        raise SystemExit(f'Policy {args.policy} not supported by spec {args.spec}')
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint_label = derive_checkpoint_label(checkpoint_path, args.checkpoint_label)
    output_root = resolve_output_root(args.output_root)
    track_tag = resolve_ordered_hse_tag(args)
    run_id = datetime.now().strftime(f'%Y%m%d_%H%M%S_{args.spec}_{track_tag}_{args.policy.lower()}')
    run_root = output_root / args.spec / track_tag / args.policy / run_id
    used_seed_root = run_root / 'used_seed_receipts'
    run_root.mkdir(parents=True, exist_ok=True)
    records: list[TaskRunRecord] = []
    for task in tasks:
        records.append(run_single_task(args=args, spec=spec, task=task, checkpoint_path=checkpoint_path, checkpoint_label=checkpoint_label, run_root=run_root, used_seed_root=used_seed_root))
    manifest = {'spec': args.spec, 'track': args.track, 'track_tag': track_tag, 'policy': args.policy, 'checkpoint': str(checkpoint_path), 'checkpoint_label': checkpoint_label, 'repo_root': str(REPO_ROOT), 'run_root': str(run_root), 'task_count': len(records), 'max_steps_per_episode': resolve_max_steps_per_episode(args, spec), 'ordered_hse_profile': args.ordered_hse_profile if args.track == 'ordered_hse' else None, 'ordered_hse': {'profile': args.ordered_hse_profile, 'backend': 'xvla', 'predictor_bundle_dir': str(resolve_checkpoint(args.real_predictor_bundle_dir)) if args.real_predictor_bundle_dir else None, 'xvla_model_dir': str(resolve_checkpoint(args.real_xvla_model_dir)) if args.real_xvla_model_dir else None, 'camera_key': args.real_camera_key, 'device': args.real_device, 'switch_mode': args.real_switch_mode, 'min_dwell_steps': args.real_min_dwell_steps, 'score_log_interval': args.real_score_log_interval, 'threshold': args.real_threshold if args.ordered_hse_profile == 'threshold' else None, 'consecutive_frames': args.real_consecutive_frames if args.ordered_hse_profile == 'threshold' else None} if args.track == 'ordered_hse' else None, 'records': [{'task_name': record.task_name, 'stage_count': record.stage_count, 'result_dir': record.result_dir, 'log_path': record.log_path, 'command_path': record.command_path, 'command': record.command, 'exit_code': record.exit_code, 'stats_path': record.stats_path, 'seed_receipt': record.seed_receipt, 'used_seed_receipt': record.used_seed_receipt} for record in records]}
    manifest_path = run_root / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    summary_cmd = [sys.executable, 'handoff_eval/summary/summarize_split.py', '--spec', args.spec, '--run-dir', str(run_root)]
    summary_result = subprocess.run(summary_cmd, cwd=REPO_ROOT, check=False)
    failed = any((record.exit_code != 0 or record.stats_path is None for record in records))
    if summary_result.returncode != 0:
        return summary_result.returncode
    return 1 if failed else 0
if __name__ == '__main__':
    raise SystemExit(main())
