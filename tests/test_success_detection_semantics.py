from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from handoff_eval.runner import run_split
from handoff_eval.summary import summarize_split

class SuccessDetectionSemanticsTest(unittest.TestCase):

    def test_run_split_uses_spec_default_max_steps(self):
        spec = run_split.load_spec('robocombine')
        task = spec['tasks'][0]
        args = SimpleNamespace(policy='Motus', track='oracle_hse', task_config=None, episodes=1, base_seed=42, save_video=False, max_steps_per_episode=None, motus_wan_dir='/tmp/fake_wan', motus_vlm_dir='/tmp/fake_vlm')
        with tempfile.TemporaryDirectory() as temp_dir:
            command = run_split.build_eval_command(args=args, spec=spec, task=task, checkpoint_path=Path('/tmp/fake_checkpoint'), checkpoint_label='fake_checkpoint', task_dir=Path(temp_dir), seed_receipt=None)
        self.assertIn('--max_steps_per_episode', command)
        max_steps_index = command.index('--max_steps_per_episode')
        self.assertEqual(command[max_steps_index + 1], '1500')

    def test_summary_reports_episode_success_rate_separately(self):
        payload = {'metadata': {'total_successes': 0, 'total_episodes': 2, 'overall_success_rate': 0.0, 'num_subtasks': 2}, 'subtask_completion': {'sequential': {'subtask_1': {'count': 2, 'rate': 1.0}, 'subtask_1_to_2': {'count': 0, 'rate': 0.0, 'conversion_from_prev': 0.0}}, 'any_order': {'all_subtasks': {'count': 0, 'rate': 0.0}}}, 'real_switch_analysis': {'predictor_switch_episode_rate': 1.0, 'premature_switch_rate': 0.5}}
        task = {'task_name': 'demo_2tasks', 'stage_count': 2}
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            (task_dir / 'evaluation_statistics.json').write_text(json.dumps(payload), encoding='utf-8')
            summary = summarize_split.summarize_task(task, task_dir)
        self.assertEqual(summary['episode_success_rate'], 0.0)
        self.assertEqual(summary['sr_ordered'], 0.0)
        self.assertEqual(summary['all_subtasks_rate'], 0.0)
        self.assertEqual(summary['predictor_switch_episode_rate'], 1.0)
        self.assertEqual(summary['premature_switch_rate'], 0.5)

    def test_ordered_hse_predictor_switch_does_not_mutate_env_completion_steps(self):
        source = Path('script/eval_policy.py').read_text(encoding='utf-8')
        self.assertNotIn('subtask_completion_steps[completed_subtask_idx] = step_count', source)
        self.assertNotIn('subtask_completion_steps[current_subtask_idx] = step_count', source)
        self.assertNotIn('subtask_completion_steps[triggered_subtask_idx] = step_count', source)
        self.assertIn('predictor_switch_steps[completed_subtask_idx] = step_count', source)
        self.assertIn('predictor_switch_steps[triggered_subtask_idx] = step_count', source)
        self.assertIn('env_subtask_completed_at_switch', source)
if __name__ == '__main__':
    unittest.main()
