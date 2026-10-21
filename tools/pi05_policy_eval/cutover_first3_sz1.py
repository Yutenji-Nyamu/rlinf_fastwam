"""Exact user-authorized old-lane stop and scheduler reservation replacement."""
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
sys.path.insert(0, str(S / "ops"))
from common import exact_signal, gpus, proc, read, same, save

D = Path(sys.argv[1])
cards = (4, 5)
assert os.getuid() == 1003
assert os.getuid() in (1003, 20001)
expected = "clean-weak-c20-sz1-v1"
active = Path("/data/chenyiteng/deployment-20261020/clean-weak-c20-sz1-v1/sweep")
assert not (D / "cutover-started.json").exists()
D.mkdir(parents=True, exist_ok=True)
plan, state, owner = read(S / "plan.json"), read(S / "status.json"), read(S / "owner-identity.json")
assert same(owner)
assert all(expected in plan["slots"][str(g)]["external_release"] for g in cards)
assert all(state["slots"][str(g)]["phase"] == "WAIT_EXTERNAL" for g in cards)
old = []
for g in cards:
    for item in (plan["slots"][str(g)]["external_identity"], read(active / f"lane-g{g}-identity.json")):
        if item not in old and same(item):
            old.append(item)
# Capture descendants of the explicit old schedulers before signalling them.
owned = {v["pid"]: v for v in old}
parents = {}
for p in Path("/proc").iterdir():
    if not p.name.isdigit():
        continue
    try:
        stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
        parents[int(p.name)] = int(stat[1])
    except (OSError, ValueError, IndexError):
        pass
changed = True
while changed:
    changed = False
    for pid, ppid in parents.items():
        if ppid in owned and pid not in owned:
            ident = proc(pid)
            if ident and ident["uid"] == os.getuid():
                owned[pid] = ident
                changed = True
before_gpu = gpus()
assert all(p["pid"] in owned for g in cards for p in before_gpu[g]["processes"]), "Unrelated target-card process"
argv = Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip("\0").split("\0")
assert str(S / "ops/owner.py") in argv
owner_env = dict(v.split("=", 1) for v in Path(f"/proc/{owner['pid']}/environ").read_text().split("\0") if "=" in v)
cwd = os.readlink(f"/proc/{owner['pid']}/cwd")
save(D / "cutover-started.json", dict(time=time.time(), plan_before=plan, state_before=state,
     old=old, owned=list(owned.values()), gpu_before=before_gpu,
     authorization="User requested stop current SZ1 Clean-weak C20 full sweep and replace with first3 takeover, then original RLT."))
for g in cards:
    save(D / f"old-g{g}-status.json", read(active / f"lane-g{g}-status.json"))
restart = False
lock = None
try:
    exact_signal(owner, signal.SIGTERM)
    for _ in range(30):
        if not same(owner):
            break
        time.sleep(1)
    assert not same(owner)
    restart = True
    lock = (S / "owner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert read(S / "plan.json") == plan
    updated = copy.deepcopy(plan)
    labels = ["lane-g4", "lane-g5"]
    identities = {}
    for label in labels:
        with (D / f"{label}-gate.log").open("x") as log:
            child = subprocess.Popen(["/usr/bin/python3", "-u", "-B", str(D / "launch_gate.py"), str(D), label],
                  cwd=D, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdin=subprocess.DEVNULL,
                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(100):
            if (D / f"{label}-identity.json").exists():
                break
            assert child.poll() is None
            time.sleep(.1)
        identities[label] = read(D / f"{label}-identity.json")
        assert same(identities[label])
    for g in cards:
        label = f"lane-g{g}"
        updated["slots"][str(g)]["external_identity"] = identities[label]
        updated["slots"][str(g)]["external_release"] = str(D / f"{label}-released.json")
    assert all(updated["slots"][g]["fallback"] == plan["slots"][g]["fallback"] for g in plan["slots"])
    assert all(updated["slots"][g] == plan["slots"][g] for g in plan["slots"] if int(g) not in cards)
    save(S / "plan.json", updated)
    save(D / "enqueued.json", dict(time=time.time(), identities=identities, cards=cards,
                                  fallbacks_preserved=True, other_slots_preserved=True))
finally:
    if lock:
        lock.close()
    if restart:
        with (D / "owner-after.log").open("x") as log:
            child = subprocess.Popen(argv, cwd=cwd, env=owner_env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(D / "owner-restarted.json", proc(child.pid))
assert (D / "enqueued.json").exists()
for ident in old:
    exact_signal(ident, signal.SIGTERM)
time.sleep(2)
for ident in owned.values():
    if same(ident):
        exact_signal(ident, signal.SIGTERM)
for _ in range(20):
    if not any(gpus()[g]["processes"] for g in cards):
        break
    time.sleep(1)
for ident in owned.values():
    if same(ident):
        exact_signal(ident, signal.SIGKILL)
time.sleep(2)
after = gpus()
assert all(not after[g]["processes"] for g in cards)
save(D / "old-stopped.json", dict(time=time.time(), intentional_user_stop=True, cards=cards,
     old_identities=old, gpu_after=after, raw_data_preserved=True))
print(json.dumps(read(D / "old-stopped.json")))
