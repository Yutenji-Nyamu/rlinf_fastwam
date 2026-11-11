import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

B = Path("/data/chenyiteng/deployment-20261111/cql-pour")
R = Path("/home/chenyiteng/results/cql-pour-20261111")
active = json.loads(
    Path(
        "/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json"
    ).read_text()
)
o = {"time": time.time(), "metrics": {}, "bindings": [], "checkpoints": {}}
for g in ["4", "6"]:
    metrics = {}
    run = Path(active["slots"][g]["run"])
    for p in (run / "formal").rglob("events.out.tfevents.*"):
        e = EventAccumulator(str(p), size_guidance={"scalars": 0})
        e.Reload()
        for k in e.Tags()["scalars"]:
            if any(
                t in k
                for t in [
                    "success_once",
                    "cql",
                    "target_q",
                    "time/",
                    "update_step",
                    "q_pi",
                    "q_data",
                    "ready_for_online",
                ]
            ):
                metrics[k] = [[v.step, v.value, v.wall_time] for v in e.Scalars(k)]
    o["metrics"][g] = metrics
    cps = sorted(
        (run / "formal/formal/checkpoints").glob("global_step_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if cps:
        p = cps[-1]
        o["checkpoints"][g] = {
            "path": str(p),
            "complete": (p / "dojo-complete.json").exists(),
            "files": [
                {"path": str(f.relative_to(p)), "bytes": f.stat().st_size}
                for f in p.rglob("*")
                if f.is_file() and (f.stat().st_size > 1024**2 or f.suffix == ".json")
            ][:25],
        }
tree = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"]))
for i, card in enumerate(tree.findall("gpu")):
    for x in card.findall("processes/process_info"):
        pid = int(x.findtext("pid"))
        path = Path("/proc") / str(pid)
        try:
            if path.stat().st_uid != 1003:
                continue
            o["bindings"].append(
                {
                    "gpu": i,
                    "pid": pid,
                    "type": x.findtext("type"),
                    "memory": x.findtext("used_memory"),
                }
            )
        except OSError:
            pass
o["disks"] = subprocess.check_output(
    ["df", "-B1", "/home/chenyiteng", "/data/chenyiteng"], text=True
)
print(json.dumps(o))
