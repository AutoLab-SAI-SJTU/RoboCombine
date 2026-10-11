from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS_ROOT = REPO_ROOT / 'handoff_eval' / 'specs'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize a RoboCombine split run.')
    parser.add_argument('--spec', required=True)
    parser.add_argument('--run-dir', required=True)
    return parser.parse_args()

def sequential_key(stage_count: int) -> str:
    if stage_count <= 1:
        return 'subtask_1'
    return 'subtask_1_to_' + '_'.join((str(index) for index in range(2, stage_count + 1)))

def conversion_key(stage_from: int, stage_to: int) -> str:
    return f's{stage_from}_to_s{stage_to}'

def mean_metric(tasks: list[dict[str, Any]], key: str) -> float | None:
    values = [task[key] for task in tasks if isinstance(task.get(key), (float, int))]
    return round(mean(values), 6) if values else None

def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def summarize_task(task: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    stats_path = task_dir / 'evaluation_statistics.json'
    if not stats_path.is_file():
        return {'task_name': task['task_name'], 'stage_count': int(task['stage_count']), 'status': 'missing', 'result_dir': str(task_dir)}
    payload = load_json(stats_path)
    stage_count = int(task['stage_count'])
    final_seq_key = sequential_key(stage_count)
    sequential = payload.get('subtask_completion', {}).get('sequential', {})
    any_order = payload.get('subtask_completion', {}).get('any_order', {})
    metadata = payload.get('metadata', {})
    final_seq = sequential.get(final_seq_key, {})
    real_switch_analysis = payload.get('real_switch_analysis') or {}
    episode_success_rate = metadata.get('overall_success_rate')
    success_count = metadata.get('total_successes')
    total_episodes = metadata.get('total_episodes')
    if episode_success_rate is None and success_count is not None and total_episodes:
        episode_success_rate = success_count / total_episodes
    stage_conversions: dict[str, float | None] = {}
    if stage_count >= 2:
        stage_conversions[conversion_key(1, 2)] = sequential.get('subtask_1_to_2', {}).get('conversion_from_prev')
    if stage_count >= 3:
        key = sequential_key(3)
        stage_conversions[conversion_key(2, 3)] = sequential.get(key, {}).get('conversion_from_prev')
    return {'task_name': task['task_name'], 'stage_count': stage_count, 'status': 'ok', 'result_dir': str(task_dir), 'stats_path': str(stats_path), 'episode_success_rate': episode_success_rate, 'sr_ordered': final_seq.get('rate'), 'all_subtasks_rate': any_order.get('all_subtasks', {}).get('rate'), 'success_count': success_count, 'total_episodes': total_episodes, 'stage_conversions': stage_conversions, 'predictor_switch_episode_rate': real_switch_analysis.get('predictor_switch_episode_rate'), 'premature_switch_rate': real_switch_analysis.get('premature_switch_rate'), 'real_switch_analysis': real_switch_analysis or None}

def render_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# Split Summary — {summary['spec']}", '', f"- hse variant: `{summary['track']}`", f"- ordered HSE profile: `{summary['ordered_hse_profile']}`" if summary.get('ordered_hse_profile') else None, f"- ordered HSE backend: `{summary.get('ordered_hse', {}).get('backend')}`" if summary.get('ordered_hse') else None, f"- policy: `{summary['policy']}`", f"- checkpoint: `{summary['checkpoint_label']}`", f"- task count: `{summary['aggregate']['task_count']}`", f"- completed tasks: `{summary['aggregate']['completed_task_count']}`", f"- mean episode success rate: `{summary['aggregate']['mean_episode_success_rate']}`", f"- mean env sr_ordered: `{summary['aggregate']['mean_sr_ordered']}`", f"- mean all_subtasks_rate: `{summary['aggregate']['mean_all_subtasks_rate']}`", '', '`episode_sr` is the native episode success rate. `env_sr_ordered` is the ordered stage-completion rate confirmed by the environment, not by an Ordered HSE predictor switch.', '', '| task | stages | episode_sr | env_sr_ordered | all_subtasks | success/episodes | ordered_hse | conversions | status |', '| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |']
    lines = [line for line in lines if line is not None]
    for task in summary['tasks']:
        conversions = task.get('stage_conversions', {})
        conversion_text = ', '.join((f'{name}={value:.3f}' if isinstance(value, (float, int)) else f'{name}=n/a' for name, value in conversions.items())) or '-'
        sr_ordered = task.get('sr_ordered')
        episode_sr = task.get('episode_success_rate')
        all_subtasks = task.get('all_subtasks_rate')
        success_count = task.get('success_count')
        total_episodes = task.get('total_episodes')
        predictor_switch_episode_rate = task.get('predictor_switch_episode_rate')
        premature_switch_rate = task.get('premature_switch_rate')
        if predictor_switch_episode_rate is None and premature_switch_rate is None:
            real_switch_text = 'n/a'
        else:
            real_switch_text = ', '.join([f'switch_ep={predictor_switch_episode_rate:.3f}' if isinstance(predictor_switch_episode_rate, (float, int)) else 'switch_ep=n/a', f'premature={premature_switch_rate:.3f}' if isinstance(premature_switch_rate, (float, int)) else 'premature=n/a'])
        lines.append('| {task_name} | {stage_count} | {episode_sr} | {sr_ordered} | {all_subtasks} | {success} | {real_switch} | {conversions} | {status} |'.format(task_name=task['task_name'], stage_count=task['stage_count'], episode_sr=f'{episode_sr:.3f}' if isinstance(episode_sr, (float, int)) else 'n/a', sr_ordered=f'{sr_ordered:.3f}' if isinstance(sr_ordered, (float, int)) else 'n/a', all_subtasks=f'{all_subtasks:.3f}' if isinstance(all_subtasks, (float, int)) else 'n/a', success=f'{success_count}/{total_episodes}' if success_count is not None and total_episodes is not None else 'n/a', real_switch=real_switch_text, conversions=conversion_text, status=task['status']))
    lines.append('')
    return '\n'.join(lines)

def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (Path.cwd() / run_dir).resolve()
    spec = load_json(SPECS_ROOT / f'{args.spec}.json')
    manifest_path = run_dir / 'run_manifest.json'
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    selected_names = None
    if manifest:
        selected_names = {record['task_name'] for record in manifest.get('records', [])}
    tasks_summary = []
    for task in spec['tasks']:
        if selected_names is not None and task['task_name'] not in selected_names:
            continue
        task_dir = run_dir / 'tasks' / task['task_name']
        tasks_summary.append(summarize_task(task, task_dir))
    completed = [task for task in tasks_summary if task['status'] == 'ok']
    aggregate = {'task_count': len(tasks_summary), 'completed_task_count': len(completed), 'mean_episode_success_rate': mean_metric(completed, 'episode_success_rate'), 'mean_sr_ordered': mean_metric(completed, 'sr_ordered'), 'mean_all_subtasks_rate': mean_metric(completed, 'all_subtasks_rate'), 'mean_predictor_switch_episode_rate': mean_metric(completed, 'predictor_switch_episode_rate'), 'mean_premature_switch_rate': mean_metric(completed, 'premature_switch_rate')}
    summary = {'spec': args.spec, 'track': manifest.get('track'), 'track_tag': manifest.get('track_tag'), 'ordered_hse_profile': manifest.get('ordered_hse_profile'), 'policy': manifest.get('policy'), 'checkpoint': manifest.get('checkpoint'), 'checkpoint_label': manifest.get('checkpoint_label'), 'ordered_hse': manifest.get('ordered_hse'), 'run_dir': str(run_dir), 'aggregate': aggregate, 'tasks': tasks_summary}
    summary_json_path = run_dir / 'summary.json'
    summary_md_path = run_dir / 'summary.md'
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    summary_md_path.write_text(render_markdown(summary) + '\n', encoding='utf-8')
    print(f'[summarize_split] wrote {summary_json_path}')
    print(f'[summarize_split] wrote {summary_md_path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
