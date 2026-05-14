import numpy as np
import torch
import dill
import os, sys
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
from pi_model import *

def encode_obs(observation):
    input_rgb_arr = [observation['observation']['head_camera']['rgb'], observation['observation']['right_camera']['rgb'], observation['observation']['left_camera']['rgb']]
    input_state = observation['joint_action']['vector']
    return (input_rgb_arr, input_state)

def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args['train_config_name'], usr_args['model_name'], usr_args['checkpoint_id'], usr_args['pi0_step'])
    custom_ckpt_path = usr_args.get('custom_ckpt_path', None)
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step, custom_ckpt_path=custom_ckpt_path)

def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)
    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    actions = model.get_action()[:model.pi0_step]
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

def reset_model(model):
    model.reset_obsrvationwindows()


def prepare_instruction_switch(
    model,
    clear_action_cache: bool = True,
    enable_forcing: bool = False,
    forcing_steps: int = 1,
    enable_cfg: bool = False,
    guidance_scale: float = 2.0,
):
    del clear_action_cache, enable_forcing, forcing_steps, enable_cfg, guidance_scale
    had_instruction = getattr(model, "instruction", None) is not None
    model.reset_obsrvationwindows()
    return {
        "cache_len_before_switch": 0,
        "prev_action_available": bool(had_instruction),
        "cache_cleared": True,
        "forcing_requested": False,
        "forcing_enabled": False,
        "cfg_enabled": False,
        "guidance_scale": None,
    }
