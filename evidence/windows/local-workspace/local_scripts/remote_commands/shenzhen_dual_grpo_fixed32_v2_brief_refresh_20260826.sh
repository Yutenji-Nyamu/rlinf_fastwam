#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
DVAC=$ROOT/dvac-global-z-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v2

TZ=Asia/Shanghai date '+snapshot=%F %T %Z'
"$VENV/bin/python" - "$CONTROL" "$DVAC" <<'PY'
from pathlib import Path
import math
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for label, root_s in zip(("control", "dvac"), sys.argv[1:]):
    root = Path(root_s)
    pid = int((root / "runtime/wrapper.pid").read_text().strip())
    alive = Path(f"/proc/{pid}").exists()
    exit_path = root / "runtime/exit_code.txt"
    exit_code = exit_path.read_text().strip() if exit_path.exists() else "pending"
    log = (root / "runtime/driver.log").read_text(errors="replace")
    fatal = sum(log.lower().count(term) for term in (
        "traceback", "outofmemory", "cuda out of memory", "nonfinite", "workercrashed"
    ))
    events = sorted((root / "tensorboard").glob("events.out.tfevents.*"))
    ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))

    def series(tag):
        return ea.Scalars(tag) if tag in tags else []

    train = series("env/success_once")
    latest = train[-1]
    last5 = sum(x.value for x in train[-5:]) / min(5, len(train))
    def last(tag):
        vals = series(tag)
        return vals[-1].value if vals else float("nan")
    evals = series("eval/success_once")
    eval_text = ",".join(f"rawstep{x.step}:{x.value:.6f}" for x in evals[-5:]) or "none"
    checkpoints = []
    for p in root.rglob("global_step_*"):
        if p.is_dir():
            try:
                checkpoints.append(int(p.name.rsplit("_", 1)[1]))
            except ValueError:
                pass
    checkpoints = sorted(set(checkpoints))
    print(
        f"{label} alive={alive} exit={exit_code} fatal_matches={fatal} "
        f"complete_global_step={latest.step + 1} train_success={latest.value:.6f} "
        f"train_success_last5={last5:.6f} kl={last('train/actor/approx_kl'):.6g} "
        f"clip={last('train/actor/clip_fraction'):.6g} grad={last('train/actor/grad_norm'):.6g} "
        f"step_seconds={last('time/step'):.3f} evals=[{eval_text}] "
        f"checkpoints={checkpoints[-5:]}"
    )
    if label == "dvac":
        print(
            f"dvac_mechanism warmup={last('train/actor/dvac_warmup'):.6g} "
            f"weight_mean={last('train/actor/dvac_weight_mean'):.6g} "
            f"ess={last('train/actor/dvac_weight_ess_fraction'):.6g}"
        )
PY

for label_run in "control:$CONTROL" "dvac:$DVAC"; do
  label=${label_run%%:*}; run=${label_run#*:}
  printf '%s_progress=' "$label"
  grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$run/runtime/driver.log" | tail -n 1 || true
  printf '%s_resource=' "$label"; tail -n 1 "$run/runtime/resource.csv"
done

awk '/^MemAvailable:/ {printf "host_mem_available_gib=%.3f\n", $2/1024/1024}' /proc/meminfo
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
