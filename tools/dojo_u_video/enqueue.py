"""One-time insertion of accepted Dojo lanes; preserve active inference and RLT."""

import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
D = Path("/data/chenyiteng/deployment-20261018/dojo-u-video-v1")
sys.path.insert(0, str(S / "ops"))
from common import exact_signal, proc, read, same, save  # noqa: E402

assert os.getuid() == 20001 and read(D / "smoke-accepted.json")["passed"]
assert not (D / "enqueued.json").exists() and not (D / "enqueue-started.json").exists()
p = read(D / "queue-prepared.json")
W = Path(p["source"])
plan = read(S / "plan.json")
state = read(S / "status.json")
owner = read(S / "owner-identity.json")
assert plan == p["plan_before"] and same(owner)
assert all(state["slots"][g]["phase"] == "WAIT_EXTERNAL" for g in ("4", "5", "6", "7"))
assert all(same(plan["slots"][g]["external_identity"]) for g in ("4", "5", "6", "7"))
args = (
    Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip("\0").split("\0")
)
assert str(S / "ops/owner.py") in args
environ = dict(
    v.split("=", 1)
    for v in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split("\0")
    if "=" in v
)
cwd = os.readlink(f"/proc/{owner['pid']}/cwd")
save(
    D / "enqueue-started.json",
    dict(
        time=time.time(),
        owner_before=owner,
        plan_before=plan,
        authorization="User: current inference -> Dojo U/full video, about 128/task -> original per-card RLT.",
    ),
)
restart = False
children = []
committed = False
lock = None
try:
    exact_signal(owner, signal.SIGTERM)
    for _ in range(50):
        if not same(owner):
            break
        time.sleep(1)
    assert not same(owner), "CPU scheduler still active"
    restart = True
    lock = (S / "owner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert read(S / "plan.json") == plan
    updated = copy.deepcopy(plan)
    status = read(S / "status.json")
    for g in ("6", "7"):
        assert same(plan["slots"][g]["external_identity"])
        with (D / f"lane-g{g}.log").open("x") as log:
            child = subprocess.Popen(
                [
                    "/usr/bin/python3",
                    "-u",
                    "-B",
                    str(W / "tools/dojo_u_video/lane.py"),
                    str(D),
                    g,
                ],
                cwd=W,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        children.append(child)
        for _ in range(100):
            if (D / f"lane-g{g}-status.json").exists():
                break
            assert child.poll() is None, "Waiting lane exited"
            time.sleep(0.1)
        ident = read(D / f"lane-g{g}-identity.json")
        assert same(ident)
        assert read(D / f"lane-g{g}-status.json")["phase"] == "WAIT_PREDECESSOR"
        updated["slots"][g]["external_identity"] = ident
        updated["slots"][g]["external_release"] = str(D / f"lane-g{g}-released.json")
        status["slots"][g] = {"phase": "WAIT_EXTERNAL"}
    assert all(
        updated["slots"][g]["fallback"] == plan["slots"][g]["fallback"]
        for g in plan["slots"]
    )
    assert all(updated["slots"][g] == plan["slots"][g] for g in ("4", "5"))
    save(S / "plan.json", updated)
    save(S / "status.json", status)
    committed = True
except BaseException:
    save(D / "enqueue-error.json", dict(time=time.time(), error=traceback.format_exc()))
    if not committed:
        for child in children:
            if child.poll() is None:
                exact_signal(proc(child.pid), signal.SIGTERM)
    raise
finally:
    if lock:
        lock.close()
    if restart:
        with (D / "scheduler-after.log").open("x") as log:
            child = subprocess.Popen(
                args,
                cwd=cwd,
                env=environ,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        save(
            D / "scheduler-restarted.json",
            dict(time=time.time(), identity=proc(child.pid)),
        )
save(
    D / "enqueued.json",
    dict(
        time=time.time(),
        gpus=[6, 7],
        current_inference_retained=all(
            same(plan["slots"][g]["external_identity"]) for g in ("6", "7")
        ),
        fallbacks_preserved=True,
        tasks=54,
        per_task=125,
        episodes=6750,
        source=str(W),
        lanes={g: updated["slots"][g]["external_identity"] for g in ("6", "7")},
    ),
)
print(json.dumps(read(D / "enqueued.json")))
