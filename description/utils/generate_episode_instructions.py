import json
import re
from typing import List, Dict, Any
import os
import argparse
import random
import yaml
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def extract_placeholders(instruction: str) -> List[str]:
    placeholders = re.findall('{([^}]+)}', instruction)
    return placeholders

def filter_instructions(instructions: List[str], episode_params: Dict[str, str]) -> List[str]:
    filtered_instructions = []
    random.shuffle(instructions)
    for instruction in instructions:
        placeholders = extract_placeholders(instruction)
        stripped_episode_params = {key.strip('{}'): value for key, value in episode_params.items()}
        arm_params = {key for key in stripped_episode_params.keys() if len(key) == 1 and 'a' <= key <= 'z'}
        non_arm_params = set(stripped_episode_params.keys()) - arm_params
        if set(placeholders) == set(stripped_episode_params.keys()) or (arm_params and set(placeholders).union(arm_params) == set(stripped_episode_params.keys()) and (not arm_params.intersection(set(placeholders)))):
            filtered_instructions.append(instruction)
    return filtered_instructions

def replace_placeholders(instruction: str, episode_params: Dict[str, str]) -> str:
    stripped_episode_params = {key.strip('{}'): value for key, value in episode_params.items()}
    for key, value in stripped_episode_params.items():
        placeholder = '{' + key + '}'
        if '\\' in value or '/' in value:
            json_path = os.path.join(os.path.join(parent_directory, '../objects_description'), value + '.json')
            if not os.path.exists(json_path):
                print(f"\x1b[1mERROR: '{json_path}' looks like a description file, but does not exist.\x1b[0m")
                exit()
        json_path = os.path.join(os.path.join(parent_directory, '../objects_description'), value + '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            description = random.choice(json_data.get('seen', []))
            value = f'the {description}'
        elif len(key) == 1 and 'a' <= key <= 'z':
            value = f'the {value} arm'
        else:
            value = f'{value}'
        instruction = instruction.replace(placeholder, value)
    return instruction

def replace_placeholders_unseen(instruction: str, episode_params: Dict[str, str]) -> str:
    stripped_episode_params = {key.strip('{}'): value for key, value in episode_params.items()}
    for key, value in stripped_episode_params.items():
        placeholder = '{' + key + '}'
        if '\\' in value or '/' in value:
            json_path = os.path.join(os.path.join(parent_directory, '../objects_description'), value + '.json')
            if not os.path.exists(json_path):
                print(f"\x1b[1mERROR: '{json_path}' looks like a description file, but does not exist.\x1b[0m")
                exit()
        json_path = os.path.join(os.path.join(parent_directory, '../objects_description'), value + '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            if 'unseen' in json_data and json_data['unseen']:
                description = random.choice(json_data.get('unseen', []))
                value = f'the {description}'
            else:
                description = random.choice(json_data.get('seen', []))
                value = f'the {description}'
        elif len(key) == 1 and 'a' <= key <= 'z':
            value = f'the {value} arm'
        else:
            value = f'{value}'
        instruction = instruction.replace(placeholder, value)
    return instruction

def load_task_instructions(task_name: str) -> Dict[str, Any]:
    file_path = os.path.join(parent_directory, f'../task_instruction/{task_name}.json')
    with open(file_path, 'r') as f:
        task_data = json.load(f)
    return task_data

def load_scene_info(task_name: str, setting: str, scene_info_path: str) -> Dict[str, Dict]:
    file_path = os.path.join(parent_directory, f'../../{scene_info_path}/{task_name}/{setting}/scene_info.json')
    try:
        with open(file_path, 'r') as f:
            scene_data = json.load(f)
        return scene_data
    except FileNotFoundError:
        print(f"\x1b[1mERROR: Scene info file '{file_path}' not found.\x1b[0m")
        exit(1)
    except json.JSONDecodeError:
        print(f"\x1b[1mERROR: Scene info file '{file_path}' contains invalid JSON.\x1b[0m")
        exit(1)

def extract_episodes_from_scene_info(scene_info: Dict) -> List[Dict[str, str]]:
    episodes = []
    for episode_key, episode_data in scene_info.items():
        if 'info' in episode_data:
            episodes.append(episode_data['info'])
        else:
            episodes.append(dict())
    return episodes

def save_episode_descriptions(task_name: str, setting: str, generated_descriptions: List[Dict]):
    output_dir = os.path.join(parent_directory, f'../../data/{task_name}/{setting}/instructions')
    os.makedirs(output_dir, exist_ok=True)
    for episode_desc in generated_descriptions:
        episode_index = episode_desc['episode_index']
        output_file = os.path.join(output_dir, f'episode{episode_index}.json')
        with open(output_file, 'w') as f:
            json.dump({'seen': episode_desc.get('seen', []), 'unseen': episode_desc.get('unseen', [])}, f, indent=2)

def generate_episode_descriptions(task_name: str, episodes: List[Dict[str, str]], max_descriptions: int=1000000):
    task_data = load_task_instructions(task_name)
    seen_instructions = task_data.get('seen', [])
    unseen_instructions = task_data.get('unseen', [])
    all_generated_descriptions = []
    for i, episode in enumerate(episodes):
        filtered_seen_instructions = filter_instructions(seen_instructions, episode)
        filtered_unseen_instructions = filter_instructions(unseen_instructions, episode)
        if filtered_seen_instructions == [] and filtered_unseen_instructions == []:
            print(f'Episode {i}: No valid instructions found')
            continue
        seen_episode_descriptions = []
        flag_seen = True
        while len(seen_episode_descriptions) < max_descriptions and flag_seen and filtered_seen_instructions:
            for instruction in filtered_seen_instructions:
                if len(seen_episode_descriptions) >= max_descriptions:
                    flag_seen = False
                    break
                description = replace_placeholders(instruction, episode)
                seen_episode_descriptions.append(description)
        unseen_episode_descriptions = []
        flag_unseen = True
        while len(unseen_episode_descriptions) < max_descriptions and flag_unseen and filtered_unseen_instructions:
            for instruction in filtered_unseen_instructions:
                if len(unseen_episode_descriptions) >= max_descriptions:
                    flag_unseen = False
                    break
                description = replace_placeholders_unseen(instruction, episode)
                unseen_episode_descriptions.append(description)
        all_generated_descriptions.append({'episode_index': i, 'seen': seen_episode_descriptions, 'unseen': unseen_episode_descriptions})
    return all_generated_descriptions
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate episode descriptions by replacing placeholders')
    parser.add_argument('task_name', type=str, help='Name of the task (JSON file name without extension)')
    parser.add_argument('setting', type=str, help='Setting name used to construct the data directory path')
    parser.add_argument('max_num', type=int, default=100, help='Maximum number of descriptions per episode')
    args = parser.parse_args()
    setting_file = os.path.join(parent_directory, f'../../task_config/{args.setting}.yml')
    with open(setting_file, 'r', encoding='utf-8') as f:
        args_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    scene_info = load_scene_info(args.task_name, args.setting, args_dict['save_path'])
    episodes = extract_episodes_from_scene_info(scene_info)
    results = generate_episode_descriptions(args.task_name, episodes, args.max_num)
    save_episode_descriptions(args.task_name, args.setting, results)
    print('Successfully Saved Instructions')
