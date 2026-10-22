# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full H50 windows from the already accepted, canonical Clean50 parquet data."""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


class DemoWindows:
    """Read each small episode once; retain compressed real RGB, not one-file churn."""

    def __init__(self, root: str, horizon: int = 50):
        self.root, self.horizon = Path(root), horizon
        self.prompts = {
            int(r["task_index"]): r["task"]
            for r in (
                json.loads(s)
                for s in (self.root / "meta/tasks.jsonl").read_text().splitlines()
                if s
            )
        }
        self.episodes, self.windows = [], []
        for path in sorted(self.root.glob("data/*/episode_*.parquet")):
            table = pq.read_table(path)
            row = {
                k: table[k].to_pylist()
                for k in (
                    "observation.state",
                    "action",
                    "task_index",
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                )
            }
            row["path"] = str(path)
            row["actions"] = np.asarray(row.pop("action"), dtype=np.float32)
            row["states"] = np.asarray(row.pop("observation.state"), dtype=np.float32)
            if row["actions"].shape != (table.num_rows, 14):
                raise ValueError(f"Expected absolute 14D Aloha actions: {path}")
            ep = len(self.episodes)
            self.episodes.append(row)
            for start in range(max(0, table.num_rows - horizon + 1)):
                if len(set(row["task_index"][start : start + horizon])) != 1:
                    raise ValueError("H50 window crosses an instruction boundary")
                self.windows.append((ep, start))
        if len(self.episodes) != 50 or not self.windows:
            raise ValueError(
                f"Expected complete Clean50; got {len(self.episodes)} episodes"
            )

    def __len__(self):
        return len(self.windows)

    def record(self, index):
        ep, start = self.windows[int(index)]
        row = self.episodes[ep]
        return row, start, self.prompts[int(row["task_index"][start])]

    def batch(self, indices):
        return decode_records((self.record(i) for i in indices), self.horizon)


def decode_records(records, horizon):
    """Accepted compressed RGB/state/absolute-action path, shared by both scopes."""
    states, actions, prompts, views = [], [], [], []
    keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    for row, start, prompt in records:
        imgs = []
        for key in keys:
            item = row[f"observation.images.{key}"][start]
            with Image.open(io.BytesIO(item["bytes"])) as image:
                imgs.append(np.asarray(image.convert("RGB")))
        views.append(np.stack(imgs))
        states.append(row["states"][start])
        actions.append(row["actions"][start : start + horizon])
        prompts.append(prompt)
    rgb = torch.from_numpy(np.stack(views))
    return {
        "main_images": rgb[:, 0],
        "wrist_images": rgb[:, 1:],
        "states": torch.from_numpy(np.stack(states)),
        "task_descriptions": prompts,
    }, torch.from_numpy(np.stack(actions))


class BalancedWindows:
    """Equal task exposure, shuffled coverage within each task, stateless resume.

    Every shuffled task cycle draws one window per task. Each task continues
    through its own shuffled window cycle, independently of budget epochs.
    No language task_index from one directory is interpreted in another.
    """

    def __init__(self, manifest, horizon=50):
        spec = json.loads(Path(manifest).read_text())
        tasks = spec["tasks"]
        names = [t["task"] for t in tasks]
        if len(set(names)) != len(names) or not tasks:
            raise ValueError("Task names must be unique and nonempty")
        self.datasets = [DemoWindows(t["path"], horizon) for t in tasks]
        self.horizon = horizon
        self.sizes = np.array([len(d) for d in self.datasets], dtype=np.int64)
        self.offsets = np.r_[0, np.cumsum(self.sizes)]
        self.manifest = [
            {
                "task": t["task"],
                "path": t["path"],
                "episodes": len(d.episodes),
                "windows": len(d),
                "instructions": len(d.prompts),
            }
            for t, d in zip(tasks, self.datasets)
        ]
        self._window_cache = {}

    def __len__(self):
        return int(self.offsets[-1])

    def indices(self, step, batch_size, seed):
        return balanced_indices(self.sizes, step, batch_size, seed, self._window_cache)

    def record(self, index):
        task = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.datasets[task].record(index - self.offsets[task])

    def batch(self, indices):
        return decode_records((self.record(i) for i in indices), self.horizon)

    def exposure(self, samples, seed):
        n = len(self.datasets)
        cycles, remainder = divmod(samples, n)
        counts = np.full(n, cycles, dtype=np.int64)
        order = np.random.default_rng(
            np.random.SeedSequence([seed, 0, cycles])
        ).permutation(n)
        counts[order[:remainder]] += 1
        return [
            {"task": m["task"], "sampled": int(k), "equivalent_epochs": float(k / size)}
            for m, k, size in zip(self.manifest, counts, self.sizes)
        ]


def balanced_indices(sizes, step, batch_size, seed, cache=None):
    sizes = np.asarray(sizes, dtype=np.int64)
    if not len(sizes) or np.any(sizes <= 0):
        raise ValueError("Every task needs at least one full action window")
    cache = {} if cache is None else cache
    n = len(sizes)
    positions = np.arange(step * batch_size, (step + 1) * batch_size)
    cycles, slots = np.divmod(positions, n)
    tasks = np.empty(batch_size, dtype=np.int64)
    for cycle in np.unique(cycles):
        select = cycles == cycle
        order = np.random.default_rng(
            np.random.SeedSequence([seed, 0, int(cycle)])
        ).permutation(n)
        tasks[select] = order[slots[select]]
    offsets = np.r_[0, np.cumsum(sizes)]
    result = np.empty(batch_size, dtype=np.int64)
    for i, (task, cycle) in enumerate(zip(tasks, cycles)):
        round_, within = divmod(int(cycle), int(sizes[task]))
        key = (seed, int(task), round_)
        if cache.get(int(task), (None,))[0] != key:
            permutation = np.random.default_rng(
                np.random.SeedSequence([seed, int(task) + 1, round_, 31337])
            ).permutation(sizes[task])
            cache[int(task)] = key, permutation
        result[i] = offsets[task] + cache[int(task)][1][within]
    return result
