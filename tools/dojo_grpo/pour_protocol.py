"""Record per-layout results and sampled, simultaneous device memory."""
import json
import subprocess
import threading
import time
from pathlib import Path


def standard_evaluation(directory, cards, after, count):
    cases = {}
    for gpu in cards:
        path = Path(directory) / f"native-g{gpu}" / "completed-waves.jsonl"
        with path.open() as stream:
            for line in stream:
                batch = json.loads(line)
                if batch["mode"] != "eval" or batch["time"] < after:
                    continue
                for row in batch["rows"]:
                    cases.setdefault(row["layout"], row)
    selected = [cases[k] for k in sorted(cases)[:count]]
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)}/{count} distinct completed evaluation layouts")
    return {"episodes": count, "successes": sum(r["success"] for r in selected),
            "rows": selected, "distinct_completed": len(cases)}


def sample_device_memory(directory, cards):
    command = ["nvidia-smi", "--id=" + ",".join(map(str, cards)),
               "--query-gpu=index,memory.used,memory.total,utilization.gpu",
               "--format=csv,noheader,nounits"]

    def collect():
        with (Path(directory) / "gpu-samples.csv").open("a", buffering=1) as stream:
            while True:
                try:
                    rows = subprocess.check_output(command, text=True, timeout=8)
                    for row in rows.splitlines():
                        stream.write(f"{time.time():.2f},{row}\n")
                except (OSError, subprocess.SubprocessError):
                    pass
                time.sleep(15)

    threading.Thread(target=collect, daemon=True).start()

