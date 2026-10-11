"""Runtime contract tests; execute real adapters/controller with fake inference backends.

AST loading skips heavyweight GPU imports only. Model weights and simulator physics
are replaced; prompt construction, switching hooks and evaluator loop are real code.
"""
import ast
from collections import deque
import logging
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

from handoff_eval.hse import (FULL_INST, ORACLE_HSE, ORDERED_HSE, HSEController,
                              InstructionSwitchConfig, parse_clear_cache)

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    namespace = dict(np=np, deque=deque, time=time, logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), path, 'exec'), namespace)
    return namespace


class Env:
    def __init__(self, completion_step=2):
        self.instruction = 'S1'
        self.take_action_cnt = 0
        self.subtask_completed = [False, False]
        self.eval_success = False
        self.actions = []
        self.completion_step = completion_step

    def get_instruction(self):
        return self.instruction

    def set_instruction(self, instruction):
        self.instruction = instruction

    def take_action(self, action, **kwargs):
        self.actions.append(int(action[0]))
        self.take_action_cnt += 1
        if self.take_action_cnt == self.completion_step:
            self.subtask_completed[0] = True

    def get_obs(self):
        return {'time': self.take_action_cnt,
                'observation': {k: {'rgb': np.zeros((2, 2, 3), dtype=np.uint8)}
                                for k in ['head_camera', 'left_camera', 'right_camera']},
                'joint_action': {'vector': np.zeros(14)}}


class Predictor:
    def __init__(self, trigger=2):
        self.frames = []
        self.resets = []
        self.trigger = trigger

    def begin_stage(self, **kwargs):
        self.resets.append(kwargs)
        return kwargs

    def observe(self, observation, env_step):
        assert observation['time'] == env_step  # never a stale pre-action frame
        self.frames.append(env_step)
        return {'should_switch': env_step == self.trigger, 'trigger_score': 0.9}


def make_policy(name):
    prompts = []
    if name == 'pi05':
        ns = load_definitions('policy/pi05/pi_model.py', ['PI0'])
        model = ns['PI0'].__new__(ns['PI0'])
        model.action_cache = deque()
        model.observation_window = None
        model.instruction = None
        model.pi0_step = 4

        def infer(obs):
            prompts.append(obs['prompt'])  # actual OpenPI inference payload
            return {'actions': np.full((4, 14), 1 if obs['prompt'] == 'S1' else 2)}

        model.policy = SimpleNamespace(infer=infer)
        ns = load_definitions('policy/pi05/deploy_policy.py', ['eval', 'encode_obs', 'reset_model'])
        return model, ns['eval'], prompts

    ns = load_definitions('policy/Motus/deploy_policy.py', ['MotusPolicy', 'eval', 'encode_obs'])
    cls = ns['MotusPolicy']
    model = cls.__new__(cls)
    model.action_cache = deque()
    import torch
    from PIL import Image
    ns.update(torch=torch, Image=Image, time=SimpleNamespace(time=time.perf_counter))
    model.obs_cache = deque()
    model.current_state = torch.zeros(1, 14)
    model.current_instruction = ''
    model.device = 'cpu'
    model.config_dict = {'model': {'inference': {'num_inference_timesteps': 1}}}
    model.inference_times = []
    model.total_inference_time = 0.0
    model.inference_count = 0
    model.step_count = 0
    model.update_obs = lambda obs: model.obs_cache.append(torch.zeros(1, 3, 2, 2))

    def encoder(texts, device):
        prompts.extend(texts)  # actual Motus text-encoder input
        return [torch.full((2, 2), 1.0 if texts[0].endswith('S1') else 2.0)]

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            text = messages[0]['content'][0]['text']
            assert text == prompts[-1]  # both language pathways agree
            return text

        def __call__(self, text, **kwargs):
            value = 1 if text[0].endswith('S1') else 2
            return {key: torch.full((1, 2), value) for key in
                    ['input_ids', 'attention_mask', 'pixel_values']}

    def inference_step(**kwargs):
        value = int(kwargs['language_embeddings'][0][0, 0])
        assert int(kwargs['vlm_inputs'][0]['input_ids'][0, 0]) == value
        return None, torch.full((1, 4, 14), float(value))

    model.t5_encoder = encoder
    model.vlm_processor = Processor()
    model.model = SimpleNamespace(inference_step=inference_step)
    return model, ns['eval'], prompts


def loop_step_namespace(env, model, eval_func, controller, state):
    # Execute the real evaluation-loop body through handoff flushing, excluding
    # console timing and the terminal break (which cannot compile outside a loop).
    tree = ast.parse((ROOT / 'script/eval_policy.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_policy')
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.While) and 'take_action_cnt' in ast.unparse(n.test))
    stop = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If) and 'step_count % 10' in ast.unparse(n.test))
    code = compile(ast.Module(body=loop.body[:stop], type_ignores=[]), 'real-eval-loop', 'exec')
    namespace = dict(time=time, traceback=traceback, task_env=env, model=model,
                     eval_func=eval_func, controller=controller, episode_state=state,
                     step_count=0, hse_variant=controller.variant,
                     ORACLE_HSE=ORACLE_HSE, ORDERED_HSE=ORDERED_HSE,
                     num_subtasks=2, subtask_completion_steps=[None, None],
                     predictor_switch_steps=state.predictor_switch_steps)
    return lambda: exec(code, namespace), namespace


@pytest.mark.parametrize('name', ['pi05', 'Motus'])
@pytest.mark.parametrize('variant', [ORACLE_HSE, ORDERED_HSE])
@pytest.mark.parametrize('clear_cache', [True, False])
@pytest.mark.parametrize('cue_step', [2, 4])
def test_language_reaches_inference_and_chunk_contract(name, variant, clear_cache, cue_step):
    env = Env(completion_step=cue_step if variant == ORACLE_HSE else 999)
    model, evaluate, prompts = make_policy(name)
    predictor = Predictor(trigger=cue_step)
    controller = HSEController(variant=variant,
        switch_config=InstructionSwitchConfig.from_args({'clear_cache': str(clear_cache).lower()}),
        stage_instructions=['S1', 'S2'], real_switch_session=predictor)
    state = controller.create_episode_state()
    controller.begin_episode(env=env, state=state, initial_instruction='S1', step_count=0)
    step, ns = loop_step_namespace(env, model, evaluate, controller, state)
    switch_at = cue_step if clear_cache else 4
    for t in range(1, switch_at + 2):
        step()
        if t < switch_at:
            assert env.instruction == 'S1'
    assert env.actions == [1] * switch_at + [2]
    assert len(prompts) == 2 and prompts[0].endswith('S1') and prompts[1].endswith('S2')
    assert len(state.switch_events) == 1
    assert state.switch_events[0]['switch_step'] == switch_at
    assert state.switch_events[0]['cache_cleared'] == clear_cache
    assert ns['subtask_completion_steps'] == ([cue_step, None] if variant == ORACLE_HSE else [None, None])
    if variant == ORDERED_HSE:
        assert state.predictor_switch_steps[0] == cue_step
        assert predictor.resets[-1]['instruction'] == 'S2'
        assert predictor.resets[-1]['episode_step'] == switch_at
        assert not any(env.subtask_completed)


@pytest.mark.parametrize('name', ['pi05', 'Motus'])
def test_full_inst_does_not_switch(name):
    env = Env()
    model, evaluate, prompts = make_policy(name)
    controller = HSEController(variant=FULL_INST, switch_config=InstructionSwitchConfig(), stage_instructions=[])
    state = controller.create_episode_state()
    env.instruction = controller.initial_instruction(full_instruction='S1 then S2')
    step, _ = loop_step_namespace(env, model, evaluate, controller, state)
    for _ in range(5):
        step()
    assert len(prompts) == 2 and all(p.endswith('S1 then S2') for p in prompts)
    assert not state.switch_events


@pytest.mark.parametrize('value, expected', [('false', False), ('False', False), (False, False), ('true', True), (True, True)])
def test_clear_cache_boolean(value, expected):
    assert parse_clear_cache(value) is expected


def test_invalid_clear_cache_rejected():
    with pytest.raises(ValueError):
        parse_clear_cache('typo')

def test_missing_switch_hook_fails_instead_of_reporting_success():
    from handoff_eval.hse import apply_instruction_switch
    with pytest.raises(TypeError):
        apply_instruction_switch(object(), config=InstructionSwitchConfig())


@pytest.mark.parametrize('clear_cache', ['true', 'false'])
def test_runner_forwards_clear_cache(clear_cache, tmp_path):
    from handoff_eval.runner import run_split
    args = SimpleNamespace(policy='pi05', track=ORACLE_HSE, task_config=None,
        episodes=1, base_seed=42, save_video=False, max_steps_per_episode=None,
        clear_cache=clear_cache, pi05_train_config='pi05_aloha_full_base',
        pi05_model_name='custom_checkpoint', pi05_checkpoint_id=0, pi05_step=50)
    spec = run_split.load_spec('robocombine')
    command = run_split.build_eval_command(args=args, spec=spec, task=spec['tasks'][0],
        checkpoint_path=tmp_path, checkpoint_label='test', task_dir=tmp_path, seed_receipt=None)
    assert command[command.index('--clear_cache') + 1] == clear_cache


@pytest.mark.parametrize('option, expected', [([], False), (['--clear_cache', 'true'], True),
                                             (['--overrides', '--clear_cache', 'false'], False)])
def test_direct_cli_and_yaml_clear_cache(option, expected, tmp_path, monkeypatch):
    import argparse
    import sys
    import yaml
    config = tmp_path / 'policy.yml'
    config.write_text('clear_cache: false\n')
    ns = load_definitions('script/eval_policy.py', ['parse_args_and_config'])
    ns.update(argparse=argparse, ast=ast, yaml=yaml, FULL_INST=FULL_INST,
              ORACLE_HSE=ORACLE_HSE, ORDERED_HSE=ORDERED_HSE, parse_clear_cache=parse_clear_cache)
    monkeypatch.setattr(sys, 'argv', ['eval_policy.py', '--config', str(config), *option])
    result = ns['parse_args_and_config']()
    assert InstructionSwitchConfig.from_args(result).clear_action_cache is expected


@pytest.mark.parametrize('name', ['pi05', 'Motus'])
def test_episode_reset_drops_old_actions_and_language(name):
    model, evaluate, prompts = make_policy(name)
    env = Env()
    evaluate(env, model, env.get_obs())
    assert len(model.action_cache) == 3
    ns = load_definitions(f'policy/{name}/deploy_policy.py', ['reset_model'])
    model.episode_count = 0
    ns['reset_model'](model)
    assert not model.action_cache
    env.instruction = 'S2'
    if name == 'Motus':
        model.current_state = __import__('torch').zeros(1, 14)
    evaluate(env, model, env.get_obs())
    assert env.actions == [1, 2]
    assert prompts[-1].endswith('S2')


@pytest.mark.parametrize('mode', ['live', 'dry_run'])
def test_threshold_predictor_rule_and_stage_reset(mode, tmp_path, monkeypatch):
    from handoff_eval import real_switch_runtime as runtime

    class Backend:
        def __init__(self, **kwargs):
            pass

        def begin_stage(self, **kwargs):
            return {}

        def observe(self, observation, env_step):
            return {'sampled': observation is not None, 'score': observation,
                    'predictor_step_index': env_step}

    monkeypatch.setattr(runtime, 'XVLARealSwitchBackend', Backend)
    monkeypatch.setattr(runtime, 'resolve_predictor_assets_dir', lambda p: p)
    session = runtime.XVLARealSwitchSession(predictor_bundle_dir=str(tmp_path),
        mode=mode, threshold=0.8, consecutive_frames=2, min_dwell_steps=3)
    session.begin_stage(instruction='S1', stage_index=0, episode_step=0, reset_reason='start')
    for step, score in enumerate([0.9, 0.7, 0.8, None, 0.9], 1):
        result = session.observe(score, env_step=step)
        assert result['should_switch'] == (step == 5 and mode == 'live')
        assert result['would_switch'] == (step == 5)
    assert len(session.export_payload()['trigger_events']) == 1
    session.begin_stage(instruction='S2', stage_index=1, episode_step=5,
                        reset_reason='switch', can_switch=False)
    for step in range(6, 10):
        assert not session.observe(0.99, env_step=step)['would_switch']
    assert session.maybe_switch()['stage_index'] == 1


@pytest.mark.parametrize('name', ['pi05', 'Motus'])
@pytest.mark.parametrize('clear_cache', ['true', 'false'])
def test_ordered_runner_forwards_threshold_and_cache(name, clear_cache, tmp_path, monkeypatch):
    import sys
    from handoff_eval.runner import run_split
    monkeypatch.setattr(sys, 'argv', ['run_split', '--spec', 'robocombine',
        '--track', ORDERED_HSE, '--policy', name, '--checkpoint', str(tmp_path),
        '--clear-cache', clear_cache, '--real-threshold', '0.85',
        '--motus-wan-dir', str(tmp_path), '--motus-vlm-dir', str(tmp_path)])
    args = run_split.parse_args()
    spec = run_split.load_spec('robocombine')
    command = run_split.build_eval_command(args=args, spec=spec, task=spec['tasks'][0],
        checkpoint_path=tmp_path, checkpoint_label='test', task_dir=tmp_path, seed_receipt=None)
    assert command[command.index('--clear_cache') + 1] == clear_cache
    assert command[command.index('--real_threshold') + 1] == '0.85'
    assert command[command.index('--real_decision_rule') + 1] == 'threshold'
