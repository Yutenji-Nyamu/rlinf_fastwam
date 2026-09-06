#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
PRISM=$ROOT/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2

TZ=Asia/Shanghai date '+snapshot=%F %T %Z'
"$VENV/bin/python" - "$CONTROL" "$PRISM" <<'PY'
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for label, root_s in zip(("control", "prism"), sys.argv[1:]):
    root = Path(root_s)
    events = sorted((root / "tensorboard").glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    pid = int((root / "runtime/wrapper.pid").read_text().strip())
    alive = Path(f"/proc/{pid}").exists()
    exit_path = root / "runtime/exit_code.txt"
    exit_code = exit_path.read_text().strip() if exit_path.exists() else "pending"
    log = (root / "runtime/driver.log").read_text(errors="replace")
    fatal_terms = ("traceback", "outofmemory", "cuda out of memory", "rayactorerror", "worker died", "errorinitializationfailed", "nccl error")
    fatal = sum(log.lower().count(term) for term in fatal_terms)
    if not events:
        print(f"{label} alive={alive} exit={exit_code} fatal={fatal} complete=none tensorboard=none")
        continue
    ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0}); ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    def series(tag): return ea.Scalars(tag) if tag in tags else []
    def last(tag):
        xs = series(tag); return xs[-1].value if xs else float("nan")
    train = series("env/success_once")
    if not train:
        print(f"{label} alive={alive} exit={exit_code} fatal={fatal} complete=none train_scalar=none")
        continue
    complete = train[-1].step + 1
    gaps = [b.wall_time-a.wall_time for a,b in zip(train[-11:-1], train[-10:])]
    step_sec = median(gaps) if gaps else last("time/step")
    remaining = max(0, 100-complete)
    eta = datetime.fromtimestamp(train[-1].wall_time + remaining*step_sec, ZoneInfo("Asia/Shanghai"))
    evals = series("eval/success_once")
    eval_text = ",".join(f"g{x.step+1}:{round(x.value*32):.0f}/32" for x in evals[-6:]) or "none"
    checkpoints = sorted({int(p.name.rsplit("_",1)[1]) for p in root.rglob("global_step_*") if p.is_dir() and p.name.rsplit("_",1)[-1].isdigit()})
    print(f"{label} alive={alive} exit={exit_code} fatal={fatal} complete={complete} raw={train[-1].value:.6f} mean5={mean(x.value for x in train[-5:]):.6f} mean10={mean(x.value for x in train[-10:]):.6f} step_sec={step_sec:.1f} eta={eta.isoformat(timespec='minutes')} evals=[{eval_text}] checkpoints={checkpoints[-4:]}")
    if label == "prism":
        prism_tags = sorted(t for t in tags if "prism" in t.lower())
        print("prism_metrics " + " ".join(f"{t}={last(t):.6g}" for t in prism_tags))
PY

for label_run in "control:$CONTROL" "prism:$PRISM"; do
  label=${label_run%%:*}; run=${label_run#*:}
  printf '%s_progress=' "$label"
  grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:|Saving checkpoint' "$run/runtime/driver.log" | tail -n1 || true
done
echo 'gpu_index,used_mib,total_mib,util_pct'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/^MemAvailable:/ {print "host_mem_available_kib=" $2}' /proc/meminfo
