import argparse
import ast
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1, encoding="utf-8")
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1, encoding="utf-8")
sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
from handoff_eval.hse import parse_clear_cache
from handoff_eval.hse import FULL_INST
from handoff_eval.hse import HSEController
from handoff_eval.hse import InstructionSwitchConfig
from handoff_eval.hse import ORDERED_HSE
from handoff_eval.hse import ORACLE_HSE
from handoff_eval.hse import resolve_hse_variant
from handoff_eval.real_switch_runtime import XVLARealSwitchSession
from envs import asset_path

CURRENT_FILE_PATH = os.path.abspath(__file__)
PARENT_DIRECTORY = os.path.dirname(CURRENT_FILE_PATH)


def _sanitize_path_component(value):
    text = str(value).strip().replace("\\", "/")
    if not text:
        return "checkpoint"
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("._")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "checkpoint"


def _resolve_output_dir(save_dir_override, default_dir):
    if not save_dir_override:
        return default_dir
    candidate = Path(str(save_dir_override)).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def apply_max_steps_override(task_env, args):
    max_steps = args.get("max_steps_per_episode")
    if max_steps in (None, "", 0):
        return
    max_steps = int(max_steps)
    if max_steps <= 0:
        return
    original_step_lim = task_env.step_lim
    task_env.step_lim = max_steps
    if original_step_lim != max_steps:
        print(f"max_steps_per_episode override: {original_step_lim} -> {max_steps}", flush=True)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as exc:
        raise SystemExit("No Task") from exc
    return env_instance


def eval_function_decorator(policy_name, model_name):
    policy_model = importlib.import_module(policy_name)
    return getattr(policy_model, model_name)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(PARENT_DIRECTORY, "../task_config/_camera_config.yml")
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as handle:
        embodiment_args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_successful_seeds(checkpoint_dir, task_name):
    checkpoint_file = Path(checkpoint_dir) / f"{task_name}_successful_seeds.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            print(f"Loaded {len(data['successful_seeds'])} successful seeds from {checkpoint_file}", flush=True)
            return set(data["successful_seeds"]), int(data.get("total_tested", 0))
        except Exception as exc:
            print(f"Warning: Failed to load checkpoint: {exc}", flush=True)
    return set(), 0


def save_successful_seeds(checkpoint_dir, task_name, successful_seeds, total_tested):
    checkpoint_file = Path(checkpoint_dir) / f"{task_name}_successful_seeds.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": task_name,
        "successful_seeds": sorted(list(successful_seeds)),
        "total_tested": int(total_tested),
        "success_rate": len(successful_seeds) / total_tested if total_tested > 0 else 0.0,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_expert_demo_seeds(checkpoint_dir, task_name):
    checkpoint_file = Path(checkpoint_dir) / f"{task_name}_expert_demo_seeds.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            print(f"Loaded {len(data['expert_demo_seeds'])} expert demo seeds from {checkpoint_file}", flush=True)
            return set(data["expert_demo_seeds"])
        except Exception as exc:
            print(f"Warning: Failed to load expert demo checkpoint: {exc}", flush=True)
    return set()


def save_expert_demo_seeds(checkpoint_dir, task_name, expert_demo_seeds):
    checkpoint_file = Path(checkpoint_dir) / f"{task_name}_expert_demo_seeds.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": task_name,
        "expert_demo_seeds": sorted(list(expert_demo_seeds)),
        "total_seeds": len(expert_demo_seeds),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_subtask_completion_patterns(checkpoint_dir, task_name):
    checkpoint_file = Path(checkpoint_dir) / f"{task_name}_subtask_patterns.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            print(f"Loaded {len(data.get('episodes', []))} episode records from {checkpoint_file}", flush=True)
            return data
        except Exception as exc:
            print(f"Warning: Failed to load subtask patterns: {exc}", flush=True)
    return {"task_name": task_name, "episodes": [], "patterns": {}, "last_updated": None}


def convert_numpy_types(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def save_subtask_completion_patterns(checkpoint_dir, task_name, patterns_data):
    checkpoint_file = Path(checkpoint_dir) / f"{task_name}_subtask_patterns.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    patterns_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_file.write_text(
        json.dumps(convert_numpy_types(patterns_data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_completion_pattern(subtask_completed, subtask_completion_steps):
    num_subtasks = len(subtask_completed)
    completed_with_steps = []
    for i in range(num_subtasks):
        if subtask_completed[i] and subtask_completion_steps[i] is not None:
            completed_with_steps.append((i + 1, subtask_completion_steps[i]))
    completed_with_steps.sort(key=lambda item: item[1])
    pattern = "".join(str(item[0]) for item in completed_with_steps)
    sequential_pattern = ""
    for i in range(num_subtasks):
        if subtask_completion_steps[i] is None:
            continue
        if i == 0:
            sequential_pattern += "1"
        elif subtask_completion_steps[i - 1] is not None and subtask_completion_steps[i] >= subtask_completion_steps[i - 1]:
            sequential_pattern += str(i + 1)
        else:
            break
    return pattern, sequential_pattern


def get_gripper_status(task_env):
    mode = getattr(task_env, "gripper_check_mode", "full")
    left_val = float(getattr(task_env.robot, "left_gripper_val", 0.0))
    right_val = float(getattr(task_env.robot, "right_gripper_val", 0.0))
    if mode == "none":
        left_open = True
        right_open = True
    elif mode == "relaxed":
        left_open = task_env.is_left_gripper_open_half() if hasattr(task_env, "is_left_gripper_open_half") else left_val > 0.6
        right_open = task_env.is_right_gripper_open_half() if hasattr(task_env, "is_right_gripper_open_half") else right_val > 0.05
    else:
        left_open = task_env.robot.is_left_gripper_open()
        right_open = task_env.robot.is_right_gripper_open()
    return {
        "gripper_check_mode": mode,
        "left_gripper_val": left_val,
        "right_gripper_val": right_val,
        "left_open": bool(left_open),
        "right_open": bool(right_open),
        "gripper_open": bool(left_open and right_open),
    }


def load_subtask_instructions(task_name, num_subtasks):
    instruction_path = os.path.join(os.path.dirname(__file__), "..", "policy", "Motus", "expr1", "task_instructions.json")
    default_instructions = ["Place cup on plate", "Place bread in basket"]
    try:
        with open(instruction_path, "r", encoding="utf-8") as handle:
            all_task_instructions = json.load(handle)
    except FileNotFoundError:
        if num_subtasks > len(default_instructions):
            raise ValueError(
                f"task_instructions.json missing and only {len(default_instructions)} fallback instructions are available for {num_subtasks} subtasks"
            )
        return default_instructions[:num_subtasks]
    task_entry = all_task_instructions.get(task_name)
    if task_entry is None:
        if num_subtasks > len(default_instructions):
            raise ValueError(
                f"No instructions found for '{task_name}' and only {len(default_instructions)} fallback instructions are available for {num_subtasks} subtasks"
            )
        return default_instructions[:num_subtasks]
    subtask_keys = sorted(
        [key for key in task_entry.keys() if key.startswith("s") and key[1:].isdigit()],
        key=lambda key: int(key[1:]),
    )
    if len(subtask_keys) < num_subtasks:
        raise ValueError(
            f"Task '{task_name}' has {len(subtask_keys)} instructions in task_instructions.json but environment requires {num_subtasks} subtasks"
        )
    return [task_entry[key] for key in subtask_keys[:num_subtasks]]


def build_run_metadata(args, hse_variant, switch_config):
    return {
        "hse_variant": hse_variant,
        "full_inst_mode": hse_variant == FULL_INST,
        "oracle_hse_mode": hse_variant == ORACLE_HSE,
        "ordered_hse_mode": hse_variant == ORDERED_HSE,
        "real_decision_rule": args.get("real_decision_rule") if hse_variant == ORDERED_HSE else None,
        "real_switch_mode": args.get("real_switch_mode") if hse_variant == ORDERED_HSE else None,
        "real_camera_key": args.get("real_camera_key") if hse_variant == ORDERED_HSE else None,
        "real_predictor_bundle_dir": args.get("real_predictor_bundle_dir") if hse_variant == ORDERED_HSE else None,
        "real_xvla_model_dir": args.get("real_xvla_model_dir") if hse_variant == ORDERED_HSE else None,
        "gripper_check_mode": args.get("gripper_check_mode", "full"),
        "max_steps_per_episode": args.get("max_steps_per_episode"),
        **switch_config.as_metadata(),
    }


def is_real_predictor_switch_event(event):
    source = str(event.get("source") or event.get("switch_source") or "")
    return source == "real_switch_predictor"


def build_real_switch_analysis(episode_details, test_num, num_subtasks):
    stage_analysis = {}
    for stage_idx in range(num_subtasks):
        stage_analysis[f"stage_{stage_idx + 1}"] = {
            "switch_count": 0,
            "env_completed_at_switch_count": 0,
            "premature_switch_count": 0,
        }
    predictor_switch_episode_count = 0
    for episode in episode_details:
        events = [event for event in episode.get("switch_events", []) if is_real_predictor_switch_event(event)]
        if events:
            predictor_switch_episode_count += 1
        for event in events:
            stage_idx = event.get("completed_subtask_idx", event.get("from_subtask_idx"))
            if stage_idx is None:
                continue
            stage_key = f"stage_{int(stage_idx) + 1}"
            if stage_key not in stage_analysis:
                continue
            env_completed = bool(event.get("env_subtask_completed_at_switch", False))
            stage_analysis[stage_key]["switch_count"] += 1
            if env_completed:
                stage_analysis[stage_key]["env_completed_at_switch_count"] += 1
            else:
                stage_analysis[stage_key]["premature_switch_count"] += 1
    total_switches = sum(stage["switch_count"] for stage in stage_analysis.values())
    premature_switch_count = sum(stage["premature_switch_count"] for stage in stage_analysis.values())
    return {
        "predictor_switch_episode_count": predictor_switch_episode_count,
        "predictor_switch_episode_rate": predictor_switch_episode_count / test_num if test_num > 0 else 0.0,
        "predictor_switch_count": total_switches,
        "premature_switch_count": premature_switch_count,
        "premature_switch_rate": premature_switch_count / total_switches if total_switches > 0 else 0.0,
        "stage_analysis": stage_analysis,
    }


def build_episode_base_record(
    task_name,
    seed,
    success,
    total_steps,
    subtask_completion_steps,
    subtask_completed,
    episode_switch_events,
    switch_defaults,
    gripper_status,
    predictor_switch_steps=None,
):
    first_switch = episode_switch_events[0] if episode_switch_events else None
    predictor_switch_steps = list(predictor_switch_steps) if predictor_switch_steps is not None else [None] * len(subtask_completed)
    first_predictor_switch = next((event for event in episode_switch_events if is_real_predictor_switch_event(event)), None)
    record = {
        "seed": seed,
        "task_name": task_name,
        "success": success,
        "episode_success": success,
        "total_steps": total_steps,
        "subtask_completion_steps": subtask_completion_steps,
        "env_subtask_completion_steps": list(subtask_completion_steps),
        "predictor_switch_steps": predictor_switch_steps,
        "predictor_switch_step": first_predictor_switch.get("switch_step") if first_predictor_switch else None,
        "switch_step": first_switch.get("switch_step") if first_switch else None,
        "s1_completed": bool(subtask_completed[0]) if len(subtask_completed) > 0 else False,
        "s2_completed": bool(subtask_completed[1]) if len(subtask_completed) > 1 else False,
        "gripper_check_mode": gripper_status["gripper_check_mode"],
        "switch_events": episode_switch_events,
        "gripper_status": gripper_status,
        "cache_len_before_switch": first_switch.get("cache_len_before_switch") if first_switch else None,
        "prev_action_before_switch": first_switch.get("prev_action_available") if first_switch else None,
    }
    for idx, step in enumerate(subtask_completion_steps, start=1):
        record[f"s{idx}_step"] = step
    for idx, step in enumerate(predictor_switch_steps, start=1):
        record[f"s{idx}_predictor_switch_step"] = step
    return record


def _build_sequential(seq_counts, test_num, num_subtasks):
    result = {}
    labels = ["subtask_1"] + ["subtask_1_to_" + "_".join(str(j) for j in range(2, i + 1)) for i in range(2, num_subtasks + 1)]
    prev = test_num
    for idx, label in enumerate(labels):
        count = seq_counts[idx]
        entry = {"count": count, "rate": count / test_num if test_num > 0 else 0.0}
        if idx > 0:
            entry["conversion_from_prev"] = count / prev if prev > 0 else 0.0
        result[label] = entry
        prev = count
    return result


def _build_funnel(seq_counts, test_num, suc_num, num_subtasks):
    result = {}
    stages = [("all", test_num, seq_counts[0])]
    for i in range(1, num_subtasks):
        stages.append((f"s{i}", seq_counts[i - 1], seq_counts[i]))
    stages.append((f"s{num_subtasks}", seq_counts[-1], suc_num))
    labels = ["all_to_s1"] + [f"s{i}_to_s{i + 1}" for i in range(1, num_subtasks)] + [f"s{num_subtasks}_to_success"]
    for label, (_, total, count) in zip(labels, stages):
        result[label] = {"count": count, "total": total, "rate": count / total if total > 0 else 0.0}
    return result


def update_evaluation_statistics(
    save_dir,
    task_name,
    policy_name,
    task_config,
    ckpt_setting,
    current_time,
    subtask_stats,
    test_num,
    suc_num,
    episode_times,
    setup_times,
    inference_times,
    eval_topic=None,
    run_metadata=None,
    num_subtasks=None,
):
    if not save_dir:
        return
    stats_file = Path(save_dir) / "evaluation_statistics.json"
    if num_subtasks is None:
        num_subtasks = 1
        while subtask_stats.get(f"subtask_{num_subtasks + 1}", 0) > 0 or bool(subtask_stats.get(f"subtask_{num_subtasks + 1}_steps")):
            num_subtasks += 1
    else:
        num_subtasks = int(num_subtasks)
    seq_counts = [subtask_stats.get("sequential_s1", 0)]
    for i in range(2, num_subtasks + 1):
        key = "sequential_s1" + "".join(f"_s{j}" for j in range(2, i + 1))
        seq_counts.append(subtask_stats.get(key, 0))
    fail_counts_by_stage = []
    prev = test_num
    for count in seq_counts:
        fail_counts_by_stage.append(prev - count)
        prev = count
    fail_after_last = prev - suc_num
    any_order = {}
    for i in range(1, num_subtasks + 1):
        key = f"subtask_{i}"
        key_steps = f"subtask_{i}_steps"
        count = subtask_stats.get(key, 0)
        steps = subtask_stats.get(key_steps, [])
        any_order[key] = {
            "count": count,
            "rate": count / test_num if test_num > 0 else 0.0,
            "avg_steps": float(np.mean(steps)) if steps else None,
            "min_steps": int(min(steps)) if steps else None,
            "max_steps": int(max(steps)) if steps else None,
        }
    any_order["all_subtasks"] = {"count": subtask_stats["all_subtasks"], "rate": subtask_stats["all_subtasks"] / test_num if test_num > 0 else 0.0}
    stats_json = {
        "metadata": {
            "eval_topic": eval_topic,
            "task_name": task_name,
            "policy_name": policy_name,
            "task_config": task_config,
            "ckpt_setting": ckpt_setting,
            "timestamp": current_time,
            "total_episodes": test_num,
            "num_subtasks": num_subtasks,
            "total_successes": suc_num,
            "overall_success_rate": suc_num / test_num if test_num > 0 else 0.0,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_metadata": run_metadata or {},
        },
        "subtask_completion": {
            "any_order": any_order,
            "sequential": _build_sequential(seq_counts, test_num, num_subtasks),
        },
        "funnel_analysis": _build_funnel(seq_counts, test_num, suc_num, num_subtasks),
        "failure_analysis": {
            "by_stage": {
                **{
                    f"failed_at_s{i + 1}": {
                        "count": fail_counts_by_stage[i],
                        "rate": fail_counts_by_stage[i] / test_num if test_num > 0 else 0.0,
                    }
                    for i in range(len(fail_counts_by_stage))
                },
                f"failed_after_s{num_subtasks}": {
                    "count": fail_after_last,
                    "rate": fail_after_last / test_num if test_num > 0 else 0.0,
                },
            },
            "by_type": {
                "timeout": {
                    "count": subtask_stats["timeout_failures"],
                    "rate": subtask_stats["timeout_failures"] / (test_num - suc_num) if test_num - suc_num > 0 else 0.0,
                },
                "gripper_not_open": {
                    "count": subtask_stats["gripper_failures"],
                    "rate": subtask_stats["gripper_failures"] / (test_num - suc_num) if test_num - suc_num > 0 else 0.0,
                },
                "subtask_incomplete": {
                    "count": subtask_stats["subtask_failures"],
                    "rate": subtask_stats["subtask_failures"] / (test_num - suc_num) if test_num - suc_num > 0 else 0.0,
                },
            },
        },
        "out_of_order": {
            "count": subtask_stats["out_of_order_episodes"],
            "rate": subtask_stats["out_of_order_episodes"] / test_num if test_num > 0 else 0.0,
        },
        "episode_details": subtask_stats["episode_details"],
        "timing": {
            "avg_episode_time": float(np.mean(episode_times)) if episode_times else None,
            "avg_setup_time": float(np.mean(setup_times)) if setup_times else None,
            "avg_inference_time": float(np.mean(inference_times)) if inference_times else None,
            "total_evaluation_time": float(sum(episode_times)) if episode_times else None,
        },
        "episode_steps": {
            "avg": float(np.mean(subtask_stats["episode_steps"])) if subtask_stats["episode_steps"] else None,
            "min": int(min(subtask_stats["episode_steps"])) if subtask_stats["episode_steps"] else None,
            "max": int(max(subtask_stats["episode_steps"])) if subtask_stats["episode_steps"] else None,
        },
    }
    if (run_metadata or {}).get("ordered_hse_mode"):
        stats_json["real_switch_analysis"] = build_real_switch_analysis(subtask_stats["episode_details"], test_num, num_subtasks)
    all_fail_counts = fail_counts_by_stage + [fail_after_last]
    max_fail_idx = all_fail_counts.index(max(all_fail_counts))
    stats_json["key_findings"] = {
        "biggest_bottleneck": f"subtask_{max_fail_idx + 1}",
        "bottleneck_rate": all_fail_counts[max_fail_idx] / test_num if test_num > 0 else 0.0,
    }
    temp_file = stats_file.with_suffix(".json.tmp")
    try:
        temp_file.write_text(json.dumps(stats_json, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_file.replace(stats_file)
    finally:
        if temp_file.exists():
            temp_file.unlink()


def _build_real_switch_session(args, hse_variant):
    if hse_variant != ORDERED_HSE:
        return None
    return XVLARealSwitchSession(
        predictor_bundle_dir=args.get("real_predictor_bundle_dir"),
        xvla_model_dir=args.get("real_xvla_model_dir"),
        camera_key=args.get("real_camera_key", "head_camera"),
        decision_rule=args.get("real_decision_rule", "threshold"),
        mode=args.get("real_switch_mode", "live"),
        threshold=args.get("real_threshold", 0.8),
        consecutive_frames=args.get("real_consecutive_frames", 3),
        min_dwell_steps=args.get("real_min_dwell_steps", 5),
        score_log_interval=args.get("real_score_log_interval", 10),
        device=args.get("real_device", "auto"),
    )


def _resolve_episode_instruction(args, task_name, episode_info, hse_variant):
    if args.get("custom_instruction"):
        return args["custom_instruction"]
    if hse_variant in {ORACLE_HSE, ORDERED_HSE}:
        return "Placeholder instruction"
    if episode_info is None:
        raise RuntimeError("Episode info is required to generate long-instruction prompts")
    episode_info_list = [episode_info["info"]]
    results = generate_episode_descriptions(args["task_name"], episode_info_list, args.get("episode_num", 1))
    return results[0]["seen"][0]


def _record_subtask_stats(task_env, subtask_stats, subtask_completion_steps, num_subtasks):
    gripper_status = get_gripper_status(task_env)
    gripper_open = gripper_status["gripper_open"]
    for i in range(num_subtasks):
        key = f"subtask_{i + 1}"
        key_steps = f"subtask_{i + 1}_steps"
        if task_env.subtask_completed[i]:
            subtask_stats[key] = subtask_stats.get(key, 0) + 1
            if subtask_completion_steps[i] is not None:
                subtask_stats.setdefault(key_steps, []).append(subtask_completion_steps[i])
    if all(task_env.subtask_completed):
        subtask_stats["all_subtasks"] += 1
    if gripper_open:
        subtask_stats["gripper_open"] += 1
    steps = [subtask_completion_steps[i] for i in range(num_subtasks)]
    is_sequential = True
    for i in range(num_subtasks):
        if steps[i] is None:
            continue
        seq_key = "sequential_" + "_".join(f"s{j + 1}" for j in range(i + 1))
        all_prev_ok = True
        for j in range(i):
            if steps[j] is None or (j > 0 and steps[j] < steps[j - 1]) or steps[i] < steps[j]:
                all_prev_ok = False
                is_sequential = False
                break
        if all_prev_ok and (i == 0 or steps[i] >= steps[i - 1]):
            subtask_stats[seq_key] = subtask_stats.get(seq_key, 0) + 1
        else:
            break
    if not is_sequential and any(step is not None for step in steps):
        subtask_stats["out_of_order_episodes"] += 1
    return gripper_status, is_sequential


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    get_model = eval_function_decorator(policy_name, "get_model")
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["policy_name"] = policy_name
    args["instruction_type"] = instruction_type
    for key in [
        "custom_instruction",
        "eval_topic",
        "episode_num",
        "max_steps_per_episode",
        "gripper_check_mode",
        "hse_variant",
        "clear_cache",
        "real_predictor_bundle_dir",
        "real_xvla_model_dir",
        "real_camera_key",
        "real_decision_rule",
        "real_switch_mode",
        "real_device",
        "real_threshold",
        "real_consecutive_frames",
        "real_min_dwell_steps",
        "real_score_log_interval",
    ]:
        if key in usr_args and usr_args.get(key) not in (None, ""):
            args[key] = usr_args[key]
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file not in (None, ""):
            return robot_file
        candidate = asset_path("embodiments", name)
        if not candidate.exists():
            raise RuntimeError(
                f"No embodiment files for {name}. "
                f"Set ROBOCOMBINE_ASSETS_DIR to an external assets root containing embodiments/{name}."
            )
        return str(candidate)

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as handle:
        camera_config = yaml.load(handle.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    ckpt_setting_label = usr_args.get("ckpt_setting_label") or _sanitize_path_component(ckpt_setting)
    default_save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting_label}/{current_time}")
    save_dir = _resolve_output_dir(usr_args.get("save_dir"), default_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_size = None
    if args["eval_video_log"]:
        camera_args = get_camera_config(args["camera"]["head_camera_type"])
        video_size = f"{camera_args['w']}x{camera_args['h']}"
        args["eval_video_save_dir"] = save_dir
    task_env = class_decorator(args["task_name"])
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    seed = usr_args["seed"]
    st_seed = 100000 * (1 + seed)
    test_num = args.get("episode_num", 100)
    log_dir = usr_args.get("log_dir")
    checkpoint_dir = log_dir if log_dir else save_dir
    model = get_model(usr_args)
    eval_policy(
        task_name,
        task_env,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        checkpoint_dir=checkpoint_dir,
        skip_successful=True,
        priority_seeds_file=usr_args.get("priority_seeds_file"),
    )
    result_path = save_dir / "_result.txt"
    result_path.write_text(f"Timestamp: {current_time}\nInstruction Type: {instruction_type}\n", encoding="utf-8")


def eval_policy(
    task_name,
    task_env,
    args,
    model,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
    checkpoint_dir=None,
    skip_successful=True,
    priority_seeds_file=None,
):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    priority_seeds = []
    if priority_seeds_file and os.path.exists(priority_seeds_file):
        priority_seeds = [int(item) for item in Path(priority_seeds_file).read_text(encoding="utf-8").split() if item.strip()]
    successful_seeds = set()
    total_previously_tested = 0
    if checkpoint_dir and skip_successful:
        successful_seeds, total_previously_tested = load_successful_seeds(checkpoint_dir, task_name)
    expert_demo_seeds = load_expert_demo_seeds(checkpoint_dir, task_name) if checkpoint_dir else set()
    subtask_patterns_data = load_subtask_completion_patterns(checkpoint_dir, task_name) if checkpoint_dir else {"task_name": task_name, "episodes": [], "patterns": {}, "last_updated": None}
    subtask_patterns_data["patterns"] = {}
    for episode in subtask_patterns_data["episodes"]:
        pattern = episode.get("pattern", "")
        if pattern:
            subtask_patterns_data["patterns"][pattern] = subtask_patterns_data["patterns"].get(pattern, 0) + 1
    task_env.suc = len(successful_seeds)
    task_env.test_num = total_previously_tested
    eval_func = eval_function_decorator(args["policy_name"], "eval")
    reset_func = eval_function_decorator(args["policy_name"], "reset_model")
    hse_variant = resolve_hse_variant(args)
    switch_config = InstructionSwitchConfig.from_args(args)
    run_metadata = build_run_metadata(args, hse_variant, switch_config)
    episode_times = []
    setup_times = []
    inference_times = []
    env_times = []
    subtask_stats = {
        "subtask_1": 0,
        "subtask_2": 0,
        "subtask_3": 0,
        "all_subtasks": 0,
        "gripper_open": 0,
        "subtask_1_steps": [],
        "subtask_2_steps": [],
        "subtask_3_steps": [],
        "episode_steps": [],
        "timeout_failures": 0,
        "gripper_failures": 0,
        "subtask_failures": 0,
        "sequential_s1": 0,
        "sequential_s1_s2": 0,
        "sequential_s1_s2_s3": 0,
        "out_of_order_episodes": 0,
        "episode_details": [],
    }
    all_seeds = [seed for seed in priority_seeds if seed not in successful_seeds]
    all_seeds.extend(seed for seed in range(st_seed, st_seed + test_num * 20) if seed not in priority_seeds and seed not in successful_seeds)
    now_id = 0
    clear_cache_freq = max(1, int(args.get("clear_cache_freq", 1) or 1))
    args["eval_mode"] = True
    seed_idx = 0
    while task_env.test_num < test_num and seed_idx < len(all_seeds):
        now_seed = all_seeds[seed_idx]
        seed_idx += 1
        episode_start_time = time.time()
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        setup_time = 0.0
        episode_info = None
        skip_expert_demo = now_seed in expert_demo_seeds
        try:
            if not skip_expert_demo:
                setup_start = time.time()
                task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                apply_max_steps_override(task_env, args)
                episode_info = task_env.play_once()
                task_env.close_env()
                setup_time = time.time() - setup_start
                setup_times.append(setup_time)
                if not (task_env.plan_success and task_env.check_success()):
                    args["render_freq"] = render_freq
                    continue
                expert_demo_seeds.add(now_seed)
                if checkpoint_dir:
                    save_expert_demo_seeds(checkpoint_dir, task_name, expert_demo_seeds)
            elif hse_variant == FULL_INST and not args.get("custom_instruction"):
                setup_start = time.time()
                task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                apply_max_steps_override(task_env, args)
                episode_info = task_env.play_once()
                task_env.close_env()
                setup_time = time.time() - setup_start
                setup_times.append(setup_time)
        except UnStableError:
            task_env.close_env()
            args["render_freq"] = render_freq
            continue
        except Exception as exc:
            task_env.close_env()
            args["render_freq"] = render_freq
            print(f"Error preparing seed {now_seed}: {exc}", flush=True)
            continue
        args["render_freq"] = render_freq
        task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        apply_max_steps_override(task_env, args)
        num_subtasks = len(task_env.subtask_completed) if hasattr(task_env, "subtask_completed") else 1
        subtask_instructions = load_subtask_instructions(args.get("task_name", task_name), num_subtasks) if hse_variant in {ORACLE_HSE, ORDERED_HSE} else []
        full_inst = _resolve_episode_instruction(args, task_name, episode_info, hse_variant)
        controller = HSEController(
            variant=hse_variant,
            switch_config=switch_config,
            stage_instructions=subtask_instructions,
            real_switch_session=_build_real_switch_session(args, hse_variant),
        )
        episode_state = controller.create_episode_state()
        initial_instruction = controller.initial_instruction(full_instruction=full_inst)
        task_env.set_instruction(instruction=initial_instruction)
        reset_event = controller.begin_episode(env=task_env, state=episode_state, initial_instruction=initial_instruction, step_count=0)
        if reset_event is not None:
            print(f"REAL predictor reset: camera={reset_event.get('camera_key')} bundle={reset_event.get('predictor_bundle_dir')}", flush=True)
        if task_env.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{task_env.eval_video_path}/episode{task_env.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            task_env._set_eval_video_ffmpeg(ffmpeg)
        reset_func(model)
        if hasattr(model, "set_task_env"):
            model.set_task_env(task_env)
        inference_start = time.time()
        step_count = 0
        succ = False
        subtask_completion_steps = [None] * num_subtasks
        predictor_switch_steps = episode_state.predictor_switch_steps
        episode_switch_events = episode_state.switch_events
        real_switch_trace_file = None
        if hse_variant == ORDERED_HSE and checkpoint_dir:
            real_switch_trace_file = os.path.join(checkpoint_dir, f"real_switch_trace_seed{now_seed}.json")
        while task_env.take_action_cnt < task_env.step_lim:
            step_start = time.time()
            observation = task_env.get_obs()
            eval_func(task_env, model, observation)
            step_count = task_env.take_action_cnt
            if hse_variant == ORDERED_HSE:
                try:
                    real_switch_metadata, switch_event = controller.observe_real_switch(
                        env=task_env,
                        model=model,
                        state=episode_state,
                        observation=task_env.get_obs(),
                        step_count=step_count,
                    )
                    if switch_event is not None:
                        completed_subtask_idx = switch_event["completed_subtask_idx"]
                        if predictor_switch_steps[completed_subtask_idx] is None:
                            predictor_switch_steps[completed_subtask_idx] = step_count
                        triggered_subtask_idx = completed_subtask_idx
                        # predictor_switch_steps[triggered_subtask_idx] = step_count
                        print(
                            f"REAL switch -> stage {switch_event['to_subtask_idx'] + 1} at step {step_count} score={real_switch_metadata.get('trigger_score')}",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"Warning: XVLA Ordered HSE inference failed at step {step_count}: {exc}", flush=True)
                    traceback.print_exc()
            if hasattr(task_env, "subtask_completed"):
                for i in range(num_subtasks):
                    if task_env.subtask_completed[i] and subtask_completion_steps[i] is None:
                        subtask_completion_steps[i] = step_count
                        if hse_variant == ORACLE_HSE and i == episode_state.current_stage_idx:
                            switch_event = controller.handle_oracle_hse(
                                env=task_env,
                                model=model,
                                state=episode_state,
                                step_count=step_count,
                            )
                            if switch_event is not None:
                                print(
                                    f"Oracle HSE -> stage {switch_event['to_subtask_idx'] + 1} at step {step_count}",
                                    flush=True,
                                )
            controller.flush_pending_switch(
                env=task_env, model=model, state=episode_state, step_count=step_count,
            )
            if step_count % 10 == 0:
                current_instruction = task_env.get_instruction() if hasattr(task_env, "get_instruction") else initial_instruction
                print(f"[Step {step_count}] Instruction={current_instruction!r} step_time={time.time() - step_start:.2f}s", flush=True)
            if task_env.eval_success:
                succ = True
                break
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        actual_steps = task_env.take_action_cnt
        subtask_stats["episode_steps"].append(actual_steps)
        if real_switch_trace_file is not None:
            trace_payload = controller.export_real_switch_trace()
            if trace_payload is not None:
                Path(real_switch_trace_file).write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        controller.close()
        if task_env.eval_video_path is not None:
            task_env._del_eval_video_ffmpeg()
        if hasattr(task_env, "subtask_completed"):
            gripper_status, is_sequential = _record_subtask_stats(task_env, subtask_stats, subtask_completion_steps, num_subtasks)
            episode_base_record = build_episode_base_record(
                task_name=task_name,
                seed=now_seed,
                success=succ,
                total_steps=actual_steps,
                subtask_completion_steps=subtask_completion_steps,
                subtask_completed=[bool(item) for item in task_env.subtask_completed],
                episode_switch_events=episode_switch_events,
                switch_defaults=switch_config.as_metadata(),
                gripper_status=gripper_status,
                predictor_switch_steps=predictor_switch_steps,
            )
            subtask_stats["episode_details"].append({**episode_base_record, "sequential": is_sequential})
            if checkpoint_dir:
                pattern, sequential_pattern = get_completion_pattern(task_env.subtask_completed, subtask_completion_steps)
                episode_record = {
                    **episode_base_record,
                    "pattern": pattern,
                    "sequential_pattern": sequential_pattern,
                    "subtask_completed": [bool(item) for item in task_env.subtask_completed],
                    "gripper_open": gripper_status["gripper_open"],
                }
                subtask_patterns_data["episodes"].append(episode_record)
                subtask_patterns_data["patterns"][pattern] = subtask_patterns_data["patterns"].get(pattern, 0) + 1
        if succ:
            task_env.suc += 1
            successful_seeds.add(now_seed)
            if checkpoint_dir:
                save_successful_seeds(checkpoint_dir, task_name, successful_seeds, task_env.test_num)
        else:
            if actual_steps >= task_env.step_lim:
                subtask_stats["timeout_failures"] += 1
            elif hasattr(task_env, "subtask_completed") and all(task_env.subtask_completed):
                if not get_gripper_status(task_env)["gripper_open"]:
                    subtask_stats["gripper_failures"] += 1
            else:
                subtask_stats["subtask_failures"] += 1
        env_cleanup_start = time.time()
        now_id += 1
        task_env.close_env(clear_cache=((task_env.test_num + 1) % clear_cache_freq == 0))
        if task_env.render_freq:
            task_env.viewer.close()
        task_env.test_num += 1
        env_times.append(time.time() - env_cleanup_start)
        episode_times.append(time.time() - episode_start_time)
        update_evaluation_statistics(
            checkpoint_dir,
            task_name,
            args["policy_name"],
            args["task_config"],
            args["ckpt_setting"],
            current_time,
            subtask_stats,
            task_env.test_num,
            task_env.suc,
            episode_times,
            setup_times,
            inference_times,
            eval_topic=args.get("eval_topic"),
            run_metadata=run_metadata,
            num_subtasks=num_subtasks,
        )
        if checkpoint_dir:
            save_subtask_completion_patterns(checkpoint_dir, task_name, subtask_patterns_data)
    if hasattr(model, "print_timing_stats"):
        model.print_timing_stats()
    return st_seed, task_env.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--hse_variant", type=str, default=FULL_INST, choices=[FULL_INST, ORACLE_HSE, ORDERED_HSE])
    parser.add_argument("--real_predictor_bundle_dir", type=str, default="")
    parser.add_argument("--real_xvla_model_dir", type=str, default="")
    parser.add_argument("--real_camera_key", type=str, default="head_camera")
    parser.add_argument("--real_decision_rule", type=str, default="threshold", choices=["threshold"])
    parser.add_argument("--real_switch_mode", type=str, default="live", choices=["live", "dry_run"])
    parser.add_argument("--real_device", type=str, default="auto")
    parser.add_argument("--real_threshold", type=float, default=0.8)
    parser.add_argument("--real_consecutive_frames", type=int, default=3)
    parser.add_argument("--real_min_dwell_steps", type=int, default=5)
    parser.add_argument("--real_score_log_interval", type=int, default=10)
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--clear_cache", type=parse_clear_cache, default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    def parse_override_pairs(pairs):
        override_dict = {}
        i = 0
        while i < len(pairs):
            key = pairs[i].lstrip("--")
            if i + 1 >= len(pairs) or pairs[i + 1].startswith("--"):
                override_dict[key] = True
                i += 1
            else:
                value = pairs[i + 1]
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    pass
                override_dict[key] = value
                i += 2
        return override_dict

    overrides = {}
    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    def set_arg_default(key, value):
        if key not in overrides:
            config[key] = value

    set_arg_default("hse_variant", args.hse_variant)
    set_arg_default("clear_cache", args.clear_cache if args.clear_cache is not None else config.get("clear_cache", True))
    set_arg_default("real_predictor_bundle_dir", args.real_predictor_bundle_dir)
    set_arg_default("real_xvla_model_dir", args.real_xvla_model_dir)
    set_arg_default("real_camera_key", args.real_camera_key)
    set_arg_default("real_decision_rule", args.real_decision_rule)
    set_arg_default("real_switch_mode", args.real_switch_mode)
    set_arg_default("real_device", args.real_device)
    set_arg_default("real_threshold", args.real_threshold)
    set_arg_default("real_consecutive_frames", args.real_consecutive_frames)
    set_arg_default("real_min_dwell_steps", args.real_min_dwell_steps)
    set_arg_default("real_score_log_interval", args.real_score_log_interval)
    if args.max_steps_per_episode is not None:
        config["max_steps_per_episode"] = args.max_steps_per_episode
    return config


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    main(usr_args)
