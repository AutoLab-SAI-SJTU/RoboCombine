from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from handoff_eval.runner import run_split


class HSEScopeTest(unittest.TestCase):
    def test_run_split_uses_hse_variant_override(self):
        spec = run_split.load_spec("robocombine")
        task = spec["tasks"][0]
        args = SimpleNamespace(
            policy="Motus",
            track="oracle_hse",
            task_config=None,
            episodes=1,
            base_seed=42,
            save_video=False,
            max_steps_per_episode=None,
            ordered_hse_profile="threshold",
            real_camera_key="head_camera",
            real_device="auto",
            real_switch_mode="live",
            real_min_dwell_steps=5,
            real_score_log_interval=10,
            real_predictor_bundle_dir=None,
            real_xvla_model_dir=None,
            real_threshold=0.8,
            real_consecutive_frames=3,
            real_ema_alpha=0.4,
            real_armed_threshold=0.82,
            real_armed_consecutive_frames=3,
            real_switch_threshold=0.9,
            real_switch_consecutive_frames=3,
            real_plateau_threshold=0.86,
            real_plateau_window=8,
            real_new_high_epsilon=0.005,
            pi05_train_config="pi05_aloha_full_base",
            pi05_model_name="custom_checkpoint",
            pi05_checkpoint_id=0,
            pi05_step=50,
            motus_wan_dir="/tmp/fake_wan",
            motus_vlm_dir="/tmp/fake_vlm",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            command = run_split.build_eval_command(
                args=args,
                spec=spec,
                task=task,
                checkpoint_path=Path("/tmp/fake_checkpoint"),
                checkpoint_label="fake_checkpoint",
                task_dir=Path(temp_dir),
                seed_receipt=None,
            )
        self.assertIn("--hse_variant", command)
        variant_index = command.index("--hse_variant")
        self.assertEqual(command[variant_index + 1], "oracle_hse")
        self.assertNotIn("--gt_instruction_mode", command)
        self.assertNotIn("--real_instruction_mode", command)
        self.assertNotIn("--real_predictor_lib_path", command)
        self.assertIn("--wan_path", command)
        self.assertIn(str(run_split.resolve_checkpoint("/tmp/fake_wan")), command)
        self.assertIn("--vlm_path", command)
        self.assertIn(str(run_split.resolve_checkpoint("/tmp/fake_vlm")), command)

    def test_pi05_prepare_instruction_switch_exists(self):
        source = Path("policy/pi05/deploy_policy.py").read_text(encoding="utf-8")
        self.assertIn("def prepare_instruction_switch(", source)
        self.assertIn("model.prepare_instruction_switch(", source)
        self.assertIn("clear_action_cache=clear_action_cache", source)


if __name__ == "__main__":
    unittest.main()
