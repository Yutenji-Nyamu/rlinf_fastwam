# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Resume migrated baseline lanes after the measured training smoke is accepted."""

import importlib.util
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def main():
    """Keep completed batches and reuse the existing lane and training lifecycle."""
    smoke = Path(sys.argv[2])
    spec = importlib.util.spec_from_file_location(
        "distillation_queue", Path(__file__).with_name("run_queue.py")
    )
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    assert os.getuid() == 20001
    queue.save(queue.D / "after-smoke-identity.json", queue.proc(os.getpid()))
    try:
        queue.event("WAIT_SELECTED_SMOKE", source=str(smoke))
        deadline = time.monotonic() + 3600
        while True:
            status_file = smoke / "selection-status.json"
            status = queue.read(status_file) if status_file.exists() else {}
            if status.get("phase") == "SELECTED_SMOKE_COMPLETE":
                break
            assert status.get("phase") != "FAILED", status
            assert time.monotonic() < deadline, (
                "Selected smoke did not finish in an hour"
            )
            time.sleep(15)
        selection = queue.read(smoke / "capacity-selected.json")
        smoke_output = Path(queue.read(smoke / "prepared.json")["output"])
        for method in ["cp", "onedp"]:
            name = selection[method]
            source = smoke_output / f"smoke-{name}"
            config = queue.read(smoke / f"{name}.json")
            resolved = queue.read(source / "resolved.json")
            assert resolved["config"] == config and config["method"] == method
            config["log_every"] = 10
            config["checkpoint_steps"] = (
                config["eval_epochs"] * resolved["steps_per_epoch"]
            )
            queue.save(queue.D / f"{method}.json", config)
            dest = queue.O / f"smoke-{method}"
            dest.mkdir()
            for fn in [
                "resolved.json",
                "metrics-rank0.jsonl",
                "metrics-rank1.jsonl",
                "inference-check-rank0.json",
                "inference-check-rank1.json",
            ]:
                shutil.copyfile(source / fn, dest / fn)
            queue.accept_smoke(method)
        queue.save(
            queue.D / "external-smoke.json",
            {
                "host": "sz2",
                "ip": "221.182.146.105",
                "passed": True,
                "time": time.time(),
                "transaction": str(smoke),
                "selection": selection,
            },
        )
        assert all(not queue.gpus()[g]["processes"] for g in [2, 3])
        # A training-only deployment may omit the evaluator's recording helpers.
        # Check the actual entrypoint once before any batch can be skipped.
        probe_env = dict(os.environ, **queue.read(queue.O / "environment-g2.json"))
        probe_env["CUDA_VISIBLE_DEVICES"] = ""
        probe = subprocess.run(
            [
                queue.PYTHON,
                str(queue.R / "tools/pi05_policy_eval/run_batch.py"),
                "--help",
            ],
            cwd=queue.R,
            env=probe_env,
            capture_output=True,
            text=True,
        )
        assert probe.returncode == 0, probe.stdout + probe.stderr
        for g in [2, 3]:
            env = dict(
                os.environ,
                **queue.read(queue.O / f"environment-g{g}.json"),
                POLICY_EVAL_ROOT=str(queue.O),
                POLICY_EVAL_TRANSACTION=str(queue.D),
                POLICY_EVAL_GPUS="2,3",
                POLICY_EVAL_UID="20001",
            )
            with (queue.D / f"lane-g{g}-migrated.log").open("x") as log:
                child = subprocess.Popen(
                    [
                        queue.PYTHON,
                        "-u",
                        "-B",
                        str(queue.R / "tools/pi05_policy_eval/lane.py"),
                        str(g),
                    ],
                    cwd=queue.R,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            identity = queue.proc(child.pid)
            queue.owned_children.append(identity)
            queue.save(queue.D / f"lane-g{g}-launch.json", identity)
        queue.main()
    except Exception:
        queue.event("FAILED", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
