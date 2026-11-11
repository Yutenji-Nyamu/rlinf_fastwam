"""One-time, CPU-only timeout correction for the measured liquid workload."""

import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

base = Path("/data/chenyiteng/deployment-20261111/cql-pour")
d = base / "g6"
script = base / "package-qa/extend_live_deadline.py"
spec = importlib.util.spec_from_file_location("bridge", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
owner = json.loads((d / "identity.json").read_text())
assert owner["pid"] == 3605114 and owner["start"] == 727707333 and module.same(owner)
assert json.loads((d / "status.json").read_text())["phase"] == "reference_eval"
plan_path = d / "plan.json"
plan = json.loads(plan_path.read_text())
original = {p["name"]: p["timeout"] for p in plan["phases"]}
assert original == {
    "reference_eval": 3600,
    "smoke": 2400,
    "smoke_resume": 2400,
    "formal": None,
}
receipt = d / "deadline-correction.json"
assert not receipt.exists()
for phase in plan["phases"]:
    if phase["timeout"] is not None:
        phase["timeout"] = 7200
temp = plan_path.with_suffix(".json.tmp")
temp.write_text(json.dumps(plan, indent=2) + "\n")
temp.replace(plan_path)
with (d / "deadline-bridge.log").open("x") as log:
    child = subprocess.Popen(
        ["/usr/bin/python3", "-u", "-B", str(script), str(d)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", DOJO_GRPO_OWNER=str(d)),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
record = {
    "time": time.time(),
    "reason": "Measured fluid rollout about 50 minutes plus 12-minute cold reset exceeds inherited 60-minute baseline guard.",
    "original_timeouts": original,
    "new_timeouts": {p["name"]: p["timeout"] for p in plan["phases"]},
    "owner": owner,
    "bridge": module.identity(child.pid),
    "training_budget_unchanged": True,
    "scope": "Only waiting CPU owner can pause near its cached old deadline; simulator/model children keep executing. Always continue owner when child exits, progress stalls, or 7200s bound is reached.",
}
receipt.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
