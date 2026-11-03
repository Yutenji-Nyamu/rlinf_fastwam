"""Small CPU-only contract shared by the simulator and RLinf processes."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np


def layout_pools(root, task):
    """Official fixed layouts: train seeds 0/2, held-out seed 1.

    Exclude exact duplicate layouts across splits, without success filtering.
    The native loader still owns asset/pose semantics.
    """
    pattern = re.compile(re.escape(task) + r"_(\d+)\.json$")
    pools = {"train": [], "eval": []}
    seen = set()
    for seed in [1, 0, 2]:
        for p in sorted((Path(root) / str(seed)).glob(task + "_*.json")):
            match = pattern.fullmatch(p.name)
            if not match:
                continue
            payload = json.loads(p.read_text())
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            pools["eval" if seed == 1 else "train"].append(
                {
                    "scene_layout": str(p),
                    "layout_seed": seed,
                    "source_layout_id": int(match[1]),
                    "digest": digest,
                }
            )
    if not all(pools.values()):
        raise ValueError("Independent train/eval native layouts are required")
    return pools


def grouped_indices(pool_size, n, group_size, rng, offset=0, fixed=False):
    if n % group_size:
        raise ValueError("A complete GRPO group must fit on each worker")
    count = n // group_size
    ids = (
        (np.arange(count) + offset) % pool_size
        if fixed
        else rng.randint(0, pool_size, size=count)
    )
    return np.repeat(ids, group_size).tolist()


def replace_invalid_layouts(ids, invalid, pool_size, group_size, lane=0):
    """Skip physical reset failures as complete groups, never policy failures."""
    result = list(ids)
    used = set(ids)
    changes = []
    # Distinct lane order avoids both eval workers choosing the same spare first.
    order = (
        list(range(32 + lane, pool_size, 2))
        + list(range(lane, pool_size, 2))
        + list(range(pool_size))
    )
    for start in range(0, len(result), group_size):
        old = result[start]
        if old not in invalid:
            continue
        replacement = next(
            (x for x in order if x not in invalid and x not in used), None
        )
        if replacement is None:
            raise RuntimeError("No unused physically valid layout candidate")
        result[start : start + group_size] = [replacement] * group_size
        used.add(replacement)
        changes.append({"group_start": start, "old": old, "replacement": replacement})
    return result, changes


def terminal_arrays(before_done, done_by_action, success_by_action):
    """Award success once, at its actual executed position; latch later dones."""
    dones = np.asarray(done_by_action, dtype=bool)
    success = np.asarray(success_by_action, dtype=bool)
    previous = np.concatenate(
        [np.asarray(before_done, dtype=bool)[:, None], dones[:, :-1]], axis=1
    )
    reward = (dones & ~previous & success).astype(np.float32)
    return reward, dones & success, dones & ~success
