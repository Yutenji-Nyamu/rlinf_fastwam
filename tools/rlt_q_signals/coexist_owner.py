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


"""Adopt only the EXPO CPU monitor; keep its training driver and return cycle.

The original four-card RLT return waits for EXPO and both higher-priority Q
experiments. No training process is restarted during this ownership transfer.
"""

import fcntl
import hashlib
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/data/chenyiteng/projects/expo-ft-sz2-20261001")
PREVIOUS = ROOT / "parallel-trial-20261006"
OLD = ROOT / "continue-60k-20261005"
TRAIN = ROOT / "formal-turn-switch-repair-20261002"
TASK = Path("/data/chenyiteng/deployment-20261006/rlt-q-signals-g67-v1")
HERE = TASK / "coexist"


def priority_released(terminals, *, boot_id, uuids):
    """Only explicit terminal plus release evidence permits the original return."""
    if set(terminals) != {"u", "norm"} or any(v is None for v in terminals.values()):
        return False
    for lane, gpu in (("u", 6), ("norm", 7)):
        row = terminals[lane]
        if (
            row.get("stage") not in ("COMPLETE", "FAILED")
            or row.get("released") is not True
        ):
            return False
        if (
            row.get("boot_id") != boot_id
            or row.get("gpu") != gpu
            or row.get("gpu_uuid") != uuids[gpu]
        ):
            raise ValueError("Priority experiment terminal identity mismatch")
        if row.get("operation_id") != "rlt-q-signals-g67-v1-" + lane:
            raise ValueError("Priority experiment operation mismatch")
    return True


def main():
    assert os.getuid() == 20001 and socket.gethostname() == "h100-gpu02"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    sys.path.insert(0, str(PREVIOUS / "source/tools/expo_parallel_20261006"))
    spec = importlib.util.spec_from_file_location(
        "previous_expo_owner", PREVIOUS / "source/tools/expo_parallel_20261006/owner.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    read, save, identity, owned = old.read, old.atomic, old.identity, old.owned
    assert HERE.is_dir() and not (HERE / "owner.json").exists()
    locks = []
    handle = (HERE / "owner.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(handle)
    initial = read(PREVIOUS / "current.json")
    assert initial["status"] == "EXPO_RUNNING" and initial["physical_gpus"] == [4, 5]
    assert time.time() - initial["time"] < 45
    assert owned(initial["owner"]) and owned(initial["child"])
    assert (
        initial["owner"]["pid"] == 18670
        and initial["owner"]["start_ticks"] == 410221234
    )
    assert not any(
        (OLD / n).exists()
        for n in ("release.json", "final.json", "rlt-resume-intent.json")
    )
    assert not (old.CYCLE / "resumed-dispatched.json").exists()
    me = identity(os.getpid())
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    uuids = old.gpu_map()
    save(
        HERE / "owner.json",
        {
            "owner": me,
            "previous": initial["owner"],
            "driver": initial["child"],
            "time": time.time(),
        },
    )
    save(HERE / "handoff-intent.json", initial)
    held = retired = False
    active = None
    expo_released = False
    passed = False
    error = None
    meta = dict(
        initial,
        owner=me,
        control=str(HERE),
        priority_control=str(TASK),
        original_control=str(PREVIOUS),
    )
    requested = [False]

    def stop(*_):
        requested[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def state(status, **more):
        meta.update(status=status, time=time.time(), **more)
        for path in (
            HERE / "current.json",
            PREVIOUS / "current.json",
            OLD / "current.json",
        ):
            save(path, meta)

    try:
        old.exact_signal(initial["owner"], signal.SIGSTOP)
        held = True
        old.wait_stopped(initial["owner"])
        assert (
            owned(initial["child"])
            and read(PREVIOUS / "current.json")["status"] == "EXPO_RUNNING"
        )
        active = old.Roster(
            initial["child"], initial["scope"], "formal", HERE / "expo-roster.json"
        )
        original_roster = read(PREVIOUS / "formal-roster.json")
        assert old.same(original_roster["root"], initial["child"])
        for row in original_roster["registered"]:
            active.rows[(row["pid"], row["start_ticks"])] = row
        active.scan()
        active.write()
        old.exact_signal(initial["owner"], signal.SIGKILL)
        retired = True
        held = False
        deadline = time.monotonic() + 10
        while owned(initial["owner"]) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not owned(initial["owner"]) and owned(initial["child"])
        for path in (
            PREVIOUS / "owner.lock",
            OLD / "owner.lock",
            TRAIN / "resource-owner.lock",
            ROOT / "eval10-continuation-20261003/resource-owner.lock",
            *[q / "owner.lock" for q in old.QUEUES],
        ):
            handle = path.open("a")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        proof = {
            "time": time.time(),
            "owner": me,
            "previous_owner": initial["owner"],
            "driver": identity(initial["child"]["pid"]),
            "driver_unchanged": True,
            "deferred_rlt_gpus": [4, 5, 6, 7],
        }
        save(HERE / "ready.json", proof)
        pointer = {
            "control": str(HERE),
            "current": str(HERE / "current.json"),
            "owner": me,
            "child": initial["child"],
            "cycle": str(old.CYCLE),
            "physical_gpus": [4, 5],
            "priority_control": str(TASK),
        }
        save(TRAIN / "active-continuation.json", pointer)
        for queue in old.QUEUES:
            save(queue / "active-continuation.json", pointer)
            save(
                queue / "queue-status.json",
                {
                    "time": time.time(),
                    "stage1": "COMPLETE",
                    "roles": {
                        "clean": "WAITING_EXPO_AND_Q_PRIORITY",
                        "combo": "WAITING_EXPO_AND_Q_PRIORITY",
                    },
                    "control": str(HERE),
                    "owner": me,
                },
            )
        while owned(initial["child"]):
            assert not requested[0], "Coordinator stop requested"
            active.scan()
            heartbeat = TRAIN / "driver-heartbeat"
            assert (
                heartbeat.is_file() and time.time() - heartbeat.stat().st_mtime < 900
            ), "EXPO heartbeat stale"
            pids = {r["pid"] for r in active.rows.values()}
            contexts = [g for g in old.gpu_rows() if g["pid"] in pids]
            assert all(g["index"] in (4, 5) for g in contexts), contexts
            state("EXPO_RUNNING", gpu_processes=contexts, q_priority="Q_U_AND_NORM")
            time.sleep(10)
        done = (
            read(TRAIN / "run/complete.json")
            if (TRAIN / "run/complete.json").exists()
            else {}
        )
        passed = bool(
            done.get("ok")
            and done.get("budget_completed")
            and done.get("cadence", {}).get("counters", {}).get("physical_actions")
            == 60000
        )
        if not passed:
            error = "Adopted EXPO driver exited without a complete 60000-action receipt"
    except BaseException:
        error = traceback.format_exc()
    finally:
        if held and owned(initial["owner"]):
            old.exact_signal(initial["owner"], signal.SIGCONT)
        if not retired:
            save(
                HERE / "handoff-failed.json",
                {"time": time.time(), "error": error, "original_owner_retained": True},
            )
            return 1
        try:
            assert active is not None
            if owned(active.root):
                active.signal(active.root, signal.SIGTERM)
                deadline = time.monotonic() + 180
                while owned(active.root) and time.monotonic() < deadline:
                    active.scan()
                    time.sleep(2)
            cleanup = active.cleanup()
            assert not any(owned(r) for r in active.rows.values())
            assert not any(g["index"] in (4, 5) for g in old.gpu_rows())
            expo_released = True
            save(
                HERE / "expo-released.json",
                {
                    "time": time.time(),
                    "expo_completed": passed,
                    "cleanup": cleanup,
                    "error": error,
                },
            )
            while True:
                terminals = {
                    lane: read(TASK / lane / "terminal.json")
                    if (TASK / lane / "terminal.json").exists()
                    else None
                    for lane in ("u", "norm")
                }
                if priority_released(terminals, boot_id=boot, uuids=uuids):
                    break
                state(
                    "WAITING_Q_PRIORITY",
                    expo_completed=passed,
                    expo_released=True,
                    error=error,
                )
                time.sleep(20)
            assert not any(g["index"] in (4, 5, 6, 7) for g in old.gpu_rows())
            # The historical return configs predate the per-card graphics scope.
            # Keep them queued until that separate binding validation is recorded.
            validation_path = HERE / "rlt-return-validation.json"
            while not validation_path.exists():
                state("WAITING_RLT_BINDING_VALIDATION", expo_released=True)
                time.sleep(20)
            validation = read(validation_path)
            assert validation["cycle"] == str(old.CYCLE)
            assert validation["boot_id"] == boot
            assert validation["compute_and_graphics_gpus"] == [4, 5, 6, 7]
            for name, digest in validation["prepared_pins"].items():
                assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
            assert validation["prepared_pins"]
            assert (
                not (OLD / "release.json").exists()
                and not (old.CYCLE / "resumed-dispatched.json").exists()
            )
            save(
                OLD / "release.json",
                {
                    "cycle_id": old.CYCLE.name,
                    "terminal_status": "completed" if passed else "failed",
                    "all_workers_stopped": True,
                    "managed_processes": list(active.rows.values()),
                    "physical_gpus": [4, 5, 6, 7],
                    "continuation_owner": me,
                    "priority_terminals": {
                        k: str(TASK / k / "terminal.json") for k in terminals
                    },
                },
            )
            state("RESTORING_RLT", expo_completed=passed, expo_released=True)
            argv = [
                old.RLT_PY,
                "-u",
                "-B",
                str(OLD / "source/tools/expo_extend_20261005/resources.py"),
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            with (HERE / "rlt-resume.log").open("x") as f:
                subprocess.run(
                    argv + ["resume"],
                    cwd=OLD / "source/tools/expo_extend_20261005",
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=300,
                )
            returned = False
            for _ in range(180):
                with (HERE / "rlt-status.log").open("w") as f:
                    subprocess.run(
                        argv + ["status"],
                        cwd=OLD / "source/tools/expo_extend_20261005",
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=90,
                    )
                if read(OLD / "rlt-status.json")["all_first_rounds_verified"]:
                    returned = True
                    break
                time.sleep(20)
            assert returned, "Original RLT first resumed rounds not verified"
            state(
                "RLT_RESTORED",
                expo_completed=passed,
                rlt_first_rounds_verified=True,
                error=error,
            )
            save(HERE / "final.json", dict(meta, expo_released=True))
        except BaseException:
            state(
                "ACTION_REQUIRED",
                error=traceback.format_exc(),
                expo_released=expo_released,
            )
            save(HERE / "final.json", meta)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
