import json
from functools import lru_cache
from pathlib import Path
from .._GLOBAL_CONFIGS import ROOT_PATH
INTERACTION_TASK_KEYWORDS = ['open', 'click', 'press', 'turn', 'switch', 'beat', 'shake', 'rotate', 'scan', 'adjust', 'grab', 'handover']
DEFAULT_ATOMIC_TIER_MANIFEST = Path(ROOT_PATH).resolve().parent / 'combination_bench' / 'manifests' / 'atomic50_task_tiers.json'

def infer_atomic_task_tier(task_name: str):
    task_lower = task_name.lower()
    if any((keyword in task_lower for keyword in INTERACTION_TASK_KEYWORDS)):
        return 'tier2_interaction'
    return 'tier1_pick_place'

@lru_cache(maxsize=4)
def load_atomic_task_tier_map(manifest_path: str | None=None):
    resolved_manifest_path = Path(manifest_path or DEFAULT_ATOMIC_TIER_MANIFEST).expanduser().resolve()
    with resolved_manifest_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload['task_tiers']

def get_atomic_task_tier(task_name: str, manifest_path: str | None=None):
    try:
        tier_map = load_atomic_task_tier_map(manifest_path)
        if task_name in tier_map:
            return tier_map[task_name]
    except FileNotFoundError:
        pass
    return infer_atomic_task_tier(task_name)
