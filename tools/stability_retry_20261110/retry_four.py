# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Recover pre-training directory-name collision using a fresh exact-owner route."""

import json
import shutil
from pathlib import Path

from prepare import BASE, SCHEDULER, read, save

entries = read(BASE / "prepared-dsrl.json")
e = entries["entries"]["4"]
old = Path(e["owner"])
assert read(old / "status.json")["phase"] == "FAILED_RETURN_RLT"
assert not (old / "smoke.log").exists() and not (old / "formal.log").exists()
state = read(SCHEDULER / "status.json")["slots"]["4"]
assert state["phase"] == "RLT_RUNNING"
new = BASE / (old.name + "-20261110-v2")
new.mkdir(exist_ok=False)
for name in ["config", "training-environment.json", "plan.json"]:
    p = old / name
    if p.is_dir():
        shutil.copytree(p, new / name)
    elif p.exists():
        shutil.copy2(p, new / name)
for p in new.rglob("*"):
    if p.is_file():
        p.write_text(p.read_text().replace(str(old), str(new)))
p = read(new / "plan.json")
p["predecessors"] = [state["identity"]]
p["pre_registered"] = False
p["from_rlt"] = True
save(new / "plan.json", p)
e.update(
    deployment=str(new),
    owner=str(new),
    old_owner=str(old),
    predecessor=state["identity"],
)
save(BASE / "prepared-dsrl-attempt1.json", read(BASE / "prepared-dsrl.json"))
save(BASE / "prepared-dsrl.json", entries)
print(json.dumps(e))
