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

"""Measure requested per-rank batches on the same authorized pair, sequentially."""

import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def main():
    """Reuse real training/checkpoint code; retain OOM evidence and test next size."""
    transaction = Path(sys.argv[1])
    while True:
        status = json.loads((transaction / "training-queue-status.json").read_text())
        if status["phase"] == "SMOKE_ONLY_COMPLETE":
            break
        if status["phase"] == "FAILED":
            raise RuntimeError("Initial B64 smoke failed; fix it before capacity tests")
        time.sleep(10)
    spec = importlib.util.spec_from_file_location(
        "distillation_queue", Path(__file__).with_name("run_queue.py")
    )
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)

    def event(phase, **fields):
        row = dict(time=time.time(), phase=phase, **fields)
        queue.save(transaction / "capacity-status.json", row)
        print(json.dumps(row), flush=True)

    queue.event = event
    queue.save(transaction / "capacity-identity.json", queue.proc(os.getpid()))
    results = []
    for method in ["cp", "onedp"]:
        base = queue.read(transaction / f"{method}.json")
        for micro in [64, 128]:
            config_name = f"{method}-m{micro}"
            config = copy.deepcopy(base)
            config.update(batch_size=micro * 2, microbatch_per_rank=micro, log_every=1)
            queue.save(transaction / f"{config_name}.json", config)
            output = queue.O / f"smoke-{config_name}"
            try:
                receipt = queue.train(
                    config_name, output, 6, label=f"capacity-{config_name}"
                )
            except RuntimeError:
                log = (transaction / f"capacity-{config_name}.log").read_text()
                if "out of memory" not in log.lower():
                    raise
                result = {
                    "method": method,
                    "micro": micro,
                    "global_batch": micro * 2,
                    "oom": True,
                }
            else:
                rows = [
                    json.loads(line)
                    for line in (output / "metrics-rank0.jsonl")
                    .read_text()
                    .splitlines()
                ]
                assert rows[-1]["step"] == 6
                tail = rows[-1]
                elapsed = (tail["time"] - rows[1]["time"]) / (
                    tail["step"] - rows[1]["step"]
                )
                result = {
                    "method": method,
                    "micro": micro,
                    "global_batch": micro * 2,
                    "oom": False,
                    "seconds_per_update": elapsed,
                    "windows_per_second": micro * 2 / elapsed,
                    "peak_gib": max(r["peak_gib"] for r in rows),
                    "peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
                    "checkpoint": receipt,
                }
            results.append(result)
            queue.save(transaction / "capacity-results.json", results)
            event("TRIAL_FINISHED", result=result)
    event("CAPACITY_COMPLETE", results=results)


if __name__ == "__main__":
    main()
