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

"""Finite per-card queue, adapted from the accepted attention/RLT lifecycle."""

import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
sys.path.insert(0, str(S / "ops"))
from common import exact_signal, gpus, proc, read, save  # noqa: E402
from recovery import archive_failed_batch, limits

R = Path(__file__).resolve().parents[2]
O = Path(os.environ["POLICY_EVAL_ROOT"])
D = Path(os.environ["POLICY_EVAL_TRANSACTION"])
g = sys.argv[1]
cards = os.environ.get("POLICY_EVAL_GPUS", "6,7").split(",")
assert len(cards) == 2 and g in cards and os.getuid() == int(os.environ.get("POLICY_EVAL_UID", "1003"))
peer = next(card for card in cards if card != g)
me = proc(os.getpid())
env = dict(
    os.environ, **read(O / f"environment-g{g}.json"), POLICY_EVAL_TOKEN=f"{D.name}-g{g}"
)
python = "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python"
child = None
save(D / f"lane-g{g}-identity.json", me)


def stop(*_):
    if child and child.poll() is None:
        exact_signal(proc(child.pid), signal.SIGTERM)
    raise SystemExit(143)


signal.signal(signal.SIGTERM, stop)


def cleanup():
    owned = []
    for card in gpus().values():
        for row in card["processes"]:
            try:
                e = dict(
                    s.split("=", 1)
                    for s in (Path("/proc") / str(row["pid"]) / "environ")
                    .read_text()
                    .split("\0")
                    if "=" in s
                )
            except OSError:
                continue
            if e.get("POLICY_EVAL_TOKEN") == env["POLICY_EVAL_TOKEN"]:
                identity = proc(row["pid"])
                assert identity and identity["uid"] == os.getuid()
                if identity not in owned:
                    owned.append(identity)
    for identity in owned:
        exact_signal(identity, signal.SIGTERM)
    if owned:
        time.sleep(2)
    assert not gpus()[int(g)]["processes"], "Assigned card is not clear"
    return {"time": time.time(), "card_clear": True, "owned_signalled": owned}


def wait_for(path, seconds=7500):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if list(D.glob("lane-g*-error.json")):
            raise RuntimeError("Peer lane failed; stop this queue and release to RLT")
        assert time.monotonic() < deadline, (
            f"Condition barrier exceeded deadline: {path}"
        )
        time.sleep(15)


try:
    plan = read(D / f"lane-g{g}-plan.json")
    budget = limits(plan)
    for stage in plan["stages"]:
        name = stage["condition"]
        for config in stage["configs"]:
            c = read(config)
            out = Path(c["output"])
            for attempt in range(budget["attempts"]):
                try:
                    assert not out.exists(), "Do not silently replay an attempted batch"
                    assert c["gpu"] == int(g)
                    if c.get("strong_weights") and c.get("takeover_mode", "weights") == "weights":
                        # Cross-host protected copies become visible by atomic rename.
                        # Reuse the finite queue wait; do not load a partial transfer.
                        save(D / f"lane-g{g}-status.json", dict(time=time.time(), phase="WAIT_WEIGHTS", config=config))
                        wait_for(Path(c["strong_weights"]), seconds=10800)
                    assert not gpus()[int(g)]["processes"]
                    record = {
                        "time": time.time(),
                        "phase": "RUNNING",
                        "config": config,
                        "gpu": int(g),
                    }
                    with (D / (out.name + f"-g{g}-attempt{attempt}.log")).open("x") as f:
                        child = subprocess.Popen(
                            [
                                python,
                                "-u",
                                "-B",
                                str(R / "tools/pi05_policy_eval/run_batch.py"),
                                config,
                            ],
                            cwd=R,
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        record["child"] = proc(child.pid)
                        save(D / f"lane-g{g}-status.json", record)
                        deadline = time.monotonic() + budget["batch_seconds"]
                        while child.poll() is None:
                            try:
                                rc = child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                progress = out / "progress.json"
                                stalled = (
                                    progress.exists()
                                    and time.time() - progress.stat().st_mtime > budget["stall_seconds"]
                                )
                                if stalled or time.monotonic() > deadline:
                                    exact_signal(record["child"], signal.SIGTERM)
                                    try:
                                        child.wait(timeout=10)
                                    except subprocess.TimeoutExpired:
                                        exact_signal(record["child"], signal.SIGKILL)
                                        child.wait(timeout=20)
                                    raise RuntimeError(
                                        f"Native batch exceeded stall={budget['stall_seconds']}s or total={budget['batch_seconds']}s"
                                    )
                        rc = child.returncode
                    clear = cleanup()
                    assert rc == 0 and (out / "done.json").exists(), (
                        f"Batch failed ({rc}); no valid completion receipt"
                    )
                    done, episodes = read(out / "done.json"), read(out / "episodes.json")
                    assert len(episodes) == done["episodes"] == 32
                    assert sum(e["success"] for e in episodes) == done["successes"]
                    assert all(e["success"] or e["submitted"] >= 200 for e in episodes)
                    save(
                        D / (out.name + "-result.json"),
                        {
                            "time": time.time(),
                            "complete": True,
                            "done": done,
                            "clearance": clear,
                        },
                    )
                    break
                except Exception:
                    if not budget["continue_errors"]:
                        raise
                    reason = traceback.format_exc()
                    clear = cleanup()
                    archive = archive_failed_batch(O, out, attempt)
                    save(D / f"{out.name}-attempt{attempt}-failed.json", dict(
                        time=time.time(), error=reason, archive=archive, clearance=clear))
            else:
                save(D / f"{out.name}-skipped.json", dict(
                    time=time.time(), attempts=budget["attempts"], complete=False))
        save(D / f"{name}-g{g}-complete.json", {"time": time.time()})
        save(
            D / f"lane-g{g}-status.json",
            {"time": time.time(), "phase": "CONDITION_BARRIER", "condition": name},
        )
        wait_for(D / f"{name}-g{peer}-complete.json", seconds=budget["barrier_seconds"])
        if g == cards[0] and stage.get("summarize", True):
            complete = all((Path(read(cfg)["output"]) / "done.json").is_file()
                           for card in cards
                           for st in read(D / f"lane-g{card}-plan.json")["stages"]
                           if st["condition"] == name for cfg in st["configs"])
            if not complete and budget["continue_errors"]:
                save(D / f"{name}-incomplete.json", dict(time=time.time(), complete=False))
                save(D / f"{name}-ready.json", dict(time=time.time(), complete=False))
            else:
                # This summarizer never creates a CUDA context.
                subprocess.run(
                    [
                        python,
                        "-B",
                        str(R / "tools/pi05_policy_eval/summarize.py"),
                        str(O),
                        name,
                    ],
                    cwd=R,
                    env=dict(env, CUDA_VISIBLE_DEVICES=""),
                    check=True,
                )
                save(D / f"{name}-ready.json", {"time": time.time()})
        elif g == cards[0]:
            save(D / f"{name}-ready.json", {"time": time.time(), "smoke_only": True})
        else:
            wait_for(D / f"{name}-ready.json", seconds=300)
    save(
        D / f"lane-g{g}-released.json",
        {
            "time": time.time(),
            "identity": me,
            "completed": not list(D.glob("*-skipped.json")),
            "queue_exhausted": True,
            "skipped_batches": [p.name for p in D.glob("*-skipped.json")],
            "clearance": cleanup(),
        },
    )
except BaseException:
    save(
        D / f"lane-g{g}-error.json",
        {"time": time.time(), "error": traceback.format_exc()},
    )
    clear = None
    try:
        clear = cleanup()
    finally:
        save(
            D / f"lane-g{g}-released.json",
            {
                "time": time.time(),
                "identity": me,
                "completed": False,
                "reason": "Queue stopped; original owner verifies exit and GPU clearance before RLT",
                "clearance": clear,
            },
        )
    raise
