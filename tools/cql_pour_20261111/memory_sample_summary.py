import csv
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

p = Path("/data/chenyiteng/deployment-20261111/cql-pour/gpu-memory.csv")
groups = {}
with p.open() as f:
    for row in csv.reader(f):
        if len(row) != 4:
            continue
        stamp, gpu, memory, util = [x.strip() for x in row]
        try:
            memory = float(memory)
            util = float(util)
        except ValueError:
            continue
        groups.setdefault(gpu, []).append((stamp, memory, util))
out = {}
for gpu, rows in groups.items():
    peak = max(rows, key=lambda x: x[1])
    out[gpu] = {
        "samples": len(rows),
        "from": rows[0][0],
        "to": rows[-1][0],
        "peak_gib": peak[1] / 1024,
        "peak_at": peak[0],
        "median_gib": statistics.median(x[1] for x in rows) / 1024,
        "latest_gib": rows[-1][1] / 1024,
        "latest_util": rows[-1][2],
    }


def summary(rows):
    if not rows:
        return {"samples": 0}
    peak = max(rows, key=lambda x: x[1])
    return {
        "samples": len(rows),
        "from": rows[0][0],
        "to": rows[-1][0],
        "peak_gib": peak[1] / 1024,
        "peak_at": peak[0],
        "median_gib": statistics.median(x[1] for x in rows) / 1024,
        "latest_gib": rows[-1][1] / 1024,
        "latest_util": rows[-1][2],
    }


def epoch(row):
    return (
        datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f")
        .replace(tzinfo=timezone(timedelta(hours=8)))
        .timestamp()
    )


status = json.loads(
    Path("/data/chenyiteng/deployment-20261111/cql-pour-56/g6/status.json").read_text()
)
cutover = datetime(
    2026, 11, 11, 19, 26, 44, tzinfo=timezone(timedelta(hours=8))
).timestamp()
result = {
    "scope": "five-second sampled device memory; includes initialization and evaluation, not steady-training averages",
    "gpus": out,
    "gpu6_56_all": summary([r for r in groups.get("6", []) if epoch(r) >= cutover]),
}
if status.get("phase") == "formal":
    started = status["started"]
    result["gpu6_56_resume"] = summary(
        [r for r in groups.get("6", []) if cutover <= epoch(r) < started]
    )
    result["gpu6_56_formal"] = summary(
        [r for r in groups.get("6", []) if epoch(r) >= started]
    )
print(json.dumps(result))
