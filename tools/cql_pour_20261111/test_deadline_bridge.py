"""CPU-only lifecycle test; no training process is touched."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

module_path = Path(
    "/data/chenyiteng/deployment-20261111/cql-pour/package-qa/extend_live_deadline.py"
)
spec = importlib.util.spec_from_file_location("bridge", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix="cql-deadline-test-") as temp:
    d = Path(temp)
    (d / "native-g6").mkdir()
    (d / "native-g6/progress.json").write_text("{}")
    owner_code = """import json,os,subprocess,time,sys
from pathlib import Path
d=Path(sys.argv[1]); child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(2)'])
def ident(pid):
 p=Path('/proc')/str(pid); f=(p/'stat').read_text().rsplit(') ',1)[1].split(); return dict(pid=pid,uid=p.stat().st_uid,start=int(f[19]))
(d/'identity.json').write_text(json.dumps(ident(os.getpid())))
(d/'reference_eval-identity.json').write_text(json.dumps(ident(child.pid)))
(d/'status.json').write_text(json.dumps(dict(phase='reference_eval',started=time.time())))
child.wait(timeout=1)
(d/'passed').write_text('child exited normally despite original one-second timeout')
(d/'status.json').write_text(json.dumps(dict(phase='formal',started=time.time())))
"""
    owner = subprocess.Popen([sys.executable, "-c", owner_code, str(d)])
    try:
        deadline = time.monotonic() + 5
        while not (d / "status.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        module.bridge(
            d, {"reference_eval": 1}, hard_limit=5, margin=0.5, poll=0.05, freshness=10
        )
        assert owner.wait(timeout=5) == 0
        assert (d / "passed").exists()
        events = [
            json.loads(line)["event"]
            for line in (d / "deadline-bridge.jsonl").read_text().splitlines()
        ]
        assert "waiting_owner_paused" in events and "owner_resumed" in events
        print(
            json.dumps(
                {
                    "passed": True,
                    "events": events,
                    "scope": "CPU dummy owner and child only",
                }
            )
        )
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
