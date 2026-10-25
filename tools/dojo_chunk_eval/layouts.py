# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Explicit episode identities over existing official Dojo layout files."""

import re
from pathlib import Path


def make_plan(root, task, episodes):
    """Use each seed directory in turn; label repeats if layouts run out."""
    pattern = re.compile(re.escape(task) + r"_(\d+)\.json$")
    groups = []
    for seed in range(3):
        rows = []
        for path in (Path(root) / str(seed)).glob("*.json"):
            match = pattern.fullmatch(path.name)
            if match:
                rows.append(
                    {
                        "layout_seed": seed,
                        "source_layout_id": int(match[1]),
                        "scene_layout": str(path),
                    }
                )
        groups.append(sorted(rows, key=lambda row: row["source_layout_id"]))
    pool = [
        group[i]
        for i in range(max(map(len, groups)))
        for group in groups
        if i < len(group)
    ]
    if not pool:
        raise ValueError(f"No native layouts for {task}")
    return [
        dict(pool[i % len(pool)], episode_id=i, repeat=i // len(pool))
        for i in range(episodes)
    ]


def install_plan(env, plan):
    """Change only the finite layout queue; native scene loading stays intact."""
    assert [row["episode_id"] for row in plan] == list(range(len(plan)))
    assert all(Path(row["scene_layout"]).is_file() for row in plan)
    manager = env.seed_manager
    manager.seed_info = {row["episode_id"]: dict(row) for row in plan}
    manager.seed_list = list(range(len(plan)))
    manager.st_idx, manager.ed_idx, manager.idx = 0, len(plan), 0
    manager._current_batch_seeds = None
