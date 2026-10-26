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

"""Bounded CPU-only decoder benchmark on representative existing parquet episodes.

This measures batch preparation, not GPU update speed or closed-loop success.
"""

import argparse
import gc
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from rlinf.algorithms.pi05_distillation.data import decode_records
from rlinf.algorithms.pi05_distillation.prefetch import decoded_batches


class EpisodeSample:
    horizon = 50

    def __init__(self, paths):
        self.rows, self.windows = [], []
        keys = ["observation.state", "action", "task_index"] + [
            "observation.images." + name
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        ]
        for path in paths:
            root = Path(path).parents[2]
            prompts = {
                int(r["task_index"]): r["task"]
                for r in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())
            }
            table = pq.read_table(path, columns=keys)
            row = {k: table[k].to_pylist() for k in keys}
            row["states"] = np.asarray(row.pop("observation.state"), dtype=np.float32)
            row["actions"] = np.asarray(row.pop("action"), dtype=np.float32)
            self.rows.append(row)
            for start in range(table.num_rows - self.horizon + 1):
                self.windows.append((row, start, prompts[int(row["task_index"][start])]))

    def batch(self, indices, *, executor=None):
        return decode_records((self.windows[int(i)] for i in indices), self.horizon, executor)


def same(a, b):
    assert torch.equal(a[1], b[1])
    for key in a[0]:
        if torch.is_tensor(a[0][key]):
            assert torch.equal(a[0][key], b[0][key])
        else:
            assert a[0][key] == b[0][key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = EpisodeSample(args.episodes)
    indices = [np.random.default_rng(814 + i).integers(len(data.windows), size=192) for i in range(4)]
    reference = data.batch(indices[0])
    results = []
    # Interleave serial and parallel trials to reduce ordering/host-load bias.
    pools = {w: ThreadPoolExecutor(w) for w in (4, 8)}
    try:
        times = {w: [] for w in (0, 4, 8)}
        for trial, order in enumerate(((0, 4, 8), (8, 0, 4), (4, 8, 0))):
            for workers in order:
                gc.collect()
                begin = time.perf_counter()
                batch = data.batch(indices[trial], executor=pools.get(workers))
                elapsed = time.perf_counter() - begin
                times[workers].append(elapsed)
                if trial == 0:
                    same(batch, reference)
                print(json.dumps({"workers": workers, "trial": trial, "seconds": elapsed}), flush=True)
                del batch
        for workers, durations in times.items():
            results.append({"workers": workers, "seconds": durations, "median_seconds": statistics.median(durations)})
    finally:
        for pool in pools.values():
            pool.shutdown()
    winner = min(results, key=lambda r: r["median_seconds"])["workers"]
    with closing(decoded_batches(data, indices, decode_workers=winner, prefetch_batches=1)) as batches:
        for i, actual in enumerate(batches):
            expected = reference if i == 0 else data.batch(indices[i])
            same(actual, expected)
            del actual, expected
    result = {
        "scope": "CPU preparation only; 192 windows, three real episodes; not full training speed",
        "episodes": args.episodes,
        "results": results,
        "selected_workers": winner,
        "prefetch_batches": 1,
        "exact_tensors_and_prompts": True,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    assert not result["cuda_initialized"]
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
