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

"""Verify full resume and exported inference at the selected measured batch."""

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path


def main():
    """Accept only the named completed trials, keeping their exact configuration."""
    transaction = Path(sys.argv[1])
    selection = json.loads((transaction / "capacity-selected.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "distillation_queue", Path(__file__).with_name("run_queue.py")
    )
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)

    def event(phase, **fields):
        row = dict(time=time.time(), phase=phase, **fields)
        queue.save(transaction / "selection-status.json", row)
        print(json.dumps(row), flush=True)

    queue.event = event
    for method in ["cp", "onedp"]:
        name = selection[method]
        config = queue.read(transaction / f"{name}.json")
        assert config["method"] == method
        output = queue.O / f"smoke-{name}"
        receipt = queue.read(output / "checkpoint.json")
        if receipt["step"] == 6:
            queue.train(
                name, output, 14, receipt["path"], label=f"capacity-resume-{name}"
            )
        else:
            assert receipt["step"] == 14
        queue.accept_smoke(name)
    event("SELECTED_SMOKE_COMPLETE", selection=selection)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (Path(sys.argv[1]) / "selection-status.json").write_text(
            json.dumps(
                {
                    "time": time.time(),
                    "phase": "FAILED",
                    "error": traceback.format_exc(),
                }
            )
        )
        raise
