"""Finite preparation gate; exec the accepted lane/sequence in this same PID."""
import json
import os
import sys
import time
from pathlib import Path

S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
sys.path.insert(0, str(S / "ops"))
from common import proc, save

D, label = Path(sys.argv[1]), sys.argv[2]
me = proc(os.getpid())
save(D / f"{label}-identity.json", me)
save(D / f"{label}-status.json", dict(time=time.time(), phase="PREPARING_REPLACEMENT"))
deadline = time.monotonic() + 14400
while time.monotonic() < deadline:
    path = D / f"{label}-launch.json"
    if path.exists():
        spec = json.loads(path.read_text())
        os.chdir(spec["cwd"])
        os.execvpe(spec["argv"][0], spec["argv"], dict(os.environ, **spec["env"]))
    time.sleep(3)
save(D / f"{label}-released.json", dict(time=time.time(), completed=False, reason="Preparation admission timeout"))
