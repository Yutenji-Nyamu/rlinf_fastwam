#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
EXPORT_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
RUNTIME=${EXPORT_ROOT}/runtime
LOG=${RUNTIME}/driver.log
CSV=${RUNTIME}/resources.csv
OUTPUT=${EXPORT_ROOT}/early_health.json
EXPECTED_HEAD=4ac48d54c63b3a83d99f551fb54f738297525acf

test ! -e "$OUTPUT"
test -f "$LOG"
test -f "$CSV"
test "$(git -C "$RLT_ROOT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$RLT_ROOT" status --porcelain)"
driver_pid=$(cat "$RUNTIME/driver_pid.txt")
kill -0 "$driver_pid"

LOG_PATH="$LOG" CSV_PATH="$CSV" OUTPUT_PATH="$OUTPUT" DRIVER_PID="$driver_pid" \
  RLT_ROOT="$RLT_ROOT" /root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
pattern = re.compile(
    r"\|\s*(?P<step>\d+)/2000 .*?"
    r"time/step=(?P<time_step>[^,\]]+), .*?"
    r"train/grad_norm=(?P<grad_norm>[^,\]]+), "
    r"train/learning_rate=(?P<learning_rate>[^,\]]+), "
    r"train/loss=(?P<loss>[^,\]]+), "
    r"train/rlt_loss=(?P<rlt_loss>[^,\]]+), "
    r"train/vla_loss=(?P<vla_loss>[^,\]]+)"
)
text = Path(os.environ["LOG_PATH"]).read_text(encoding="utf-8", errors="replace")
clean = ansi.sub("", text).replace("\r", "\n")
by_step = {}
for line in clean.splitlines():
    match = pattern.search(line)
    if match is None:
        continue
    payload = {"step": int(match.group("step"))}
    for key in (
        "time_step",
        "grad_norm",
        "learning_rate",
        "loss",
        "rlt_loss",
        "vla_loss",
    ):
        payload[key] = float(match.group(key))
        if not math.isfinite(payload[key]):
            raise ValueError(f"non-finite {key} at step {payload['step']}")
    if payload["loss"] != payload["rlt_loss"] or payload["vla_loss"] != 0.0:
        raise ValueError(f"loss contract mismatch at step {payload['step']}: {payload}")
    by_step[payload["step"]] = payload
if not by_step or max(by_step) < 20:
    raise ValueError(f"not enough early metrics: latest={max(by_step) if by_step else None}")

selected_steps = [step for step in (1, 2, 5, 10, 20, 50) if step in by_step]
latest_step = max(by_step)
if latest_step not in selected_steps:
    selected_steps.append(latest_step)
selected = [by_step[step] for step in selected_steps]
recent = [by_step[step]["time_step"] for step in sorted(by_step)[-20:]]
steady_step_seconds = sorted(recent)[len(recent) // 2]

resource_rows = []
with Path(os.environ["CSV_PATH"]).open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        resource_rows.append(row)
if not resource_rows:
    raise ValueError("resource CSV has no samples")
max_gpu0 = max(int(row["gpu0_used_mib"]) for row in resource_rows)
max_gpu1 = max(int(row["gpu1_used_mib"]) for row in resource_rows)
max_rss_kib = max(int(row["matched_rss_kib"]) for row in resource_rows)

error_patterns = {
    "oom": r"out of memory",
    "cuda_error": r"CUDA error",
    "traceback": r"Traceback",
    "nccl_error": r"NCCL.*error",
    "child_failed": r"ChildFailed",
    "killed": r"\bkilled\b",
}
errors = {
    name: len(re.findall(expression, clean, flags=re.IGNORECASE))
    for name, expression in error_patterns.items()
}
if any(errors.values()):
    raise ValueError(f"early error signals: {errors}")

gpu_text = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
gpu_now = [
    {
        "index": int(parts[0].strip()),
        "memory_used_mib": int(parts[1].strip()),
        "utilization_pct": int(parts[2].strip()),
    }
    for line in gpu_text.splitlines()
    if (parts := line.split(","))
]

payload = {
    "observed_at": datetime.now().astimezone().isoformat(),
    "state": "running",
    "driver_pid": int(os.environ["DRIVER_PID"]),
    "git_head": subprocess.check_output(
        ["git", "-C", os.environ["RLT_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "latest_step": latest_step,
    "total_steps": 2000,
    "selected_metrics": selected,
    "steady_step_seconds_median_last20": steady_step_seconds,
    "estimated_remaining_seconds": (2000 - latest_step) * steady_step_seconds,
    "resource_samples": len(resource_rows),
    "max_gpu0_used_mib": max_gpu0,
    "max_gpu1_used_mib": max_gpu1,
    "max_matched_rss_kib": max_rss_kib,
    "gpu_now": gpu_now,
    "error_counts": errors,
    "checkpoint_expected_only_at_step": 2000,
}
Path(os.environ["OUTPUT_PATH"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

sha256sum "$OUTPUT" > "${OUTPUT}.sha256"
