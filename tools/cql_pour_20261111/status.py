import json
import subprocess
import time
from pathlib import Path

B = Path("/data/chenyiteng/deployment-20261111/cql-pour")
active = json.loads(
    Path(
        "/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json"
    ).read_text()
)
o = {"time": time.time(), "slots": {}}
for g in ["4", "6"]:
    slot = active["slots"][g]
    d = Path(slot["deployment"])
    r = {"deployment": str(d), "run": slot["run"]}
    p = json.loads((d / "plan.json").read_text())
    r["num_envs"] = p["num_envs"]
    for f in [
        "status.json",
        "gpu-smoke.json",
        "reference-eval.json",
        "binding.json",
        "identity.json",
    ]:
        if (d / f).exists():
            r[f] = json.loads((d / f).read_text())
    phase = r.get("status.json", {}).get("phase", "owner")
    f = d / (phase + ".log")
    if f.exists():
        with f.open("rb") as s:
            s.seek(max(0, s.seek(0, 2) - 8000))
            r["tail"] = s.read().decode(errors="replace")[-5000:]
    run = Path(slot["run"])
    if (run / "stage1/complete.json").exists():
        r["stage1"] = json.loads((run / "stage1/complete.json").read_text())
    r["complete_checkpoints"] = [
        str(x)
        for x in run.glob("formal/formal/checkpoints/global_step_*/dojo-complete.json")
    ]
    o["slots"][g] = r
o["gpu"] = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ],
    text=True,
)
o["processes"] = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader",
    ],
    text=True,
)
print(json.dumps(o))
