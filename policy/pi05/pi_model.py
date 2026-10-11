from collections import deque
import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import os

class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, custom_ckpt_path=None):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        if custom_ckpt_path:
            checkpoint_base = custom_ckpt_path
            assets_path = os.path.join(custom_ckpt_path, 'assets/')
            if not os.path.exists(assets_path):
                raise FileNotFoundError(f'Assets path not found: {assets_path}')
            entries = os.listdir(assets_path)
            assets_id = entries[0] if entries else ''
        else:
            raise FileNotFoundError(
                'pi05 checkpoints are not bundled with RoboCombine. '
                'Pass custom_ckpt_path pointing to an external checkpoint directory.'
            )
        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(config, checkpoint_base, robotwin_repo_id=assets_id)
        print(f'loading model success from: {checkpoint_base}')
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.action_cache = deque()

    def set_img_size(self, img_size):
        self.img_size = img_size

    def set_language(self, instruction):
        self.instruction = instruction
        print(f'successfully set instruction:{instruction}')

    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (img_arr[0], img_arr[1], img_arr[2], state)
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))
        self.observation_window = {'state': state, 'images': {'cam_high': img_front, 'cam_left_wrist': img_left, 'cam_right_wrist': img_right}, 'prompt': self.instruction}

    def get_action(self):
        assert self.observation_window is not None, 'update observation_window first!'
        return self.policy.infer(self.observation_window)['actions']

    def prepare_instruction_switch(self, clear_action_cache=True, **kwargs):
        cache_len = len(self.action_cache)
        if clear_action_cache:
            self.action_cache.clear()
        self.instruction = None
        self.observation_window = None
        return {"cache_len_before_switch": cache_len,
                "cache_cleared": bool(clear_action_cache),
                "forcing_requested": False, "forcing_enabled": False, "cfg_enabled": False}

    def reset_obsrvationwindows(self):
        self.action_cache.clear()
        self.instruction = None
        self.observation_window = None
        print('successfully unset obs and language intruction')
