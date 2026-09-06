#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
PRISM=$ROOT/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1

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
    ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))

    def series(tag):
        return ea.Scalars(tag) if tag in tags else []

    def last(tag):
        values = series(tag)
        return values[-1].value if values else float("nan")

    train = series("env/success_once")
    latest = train[-1]
    complete = latest.step + 1
    recent_gaps = [b.wall_time - a.wall_time for a, b in zip(train[-11:-1], train[-10:])]
    step_sec = median(recent_gaps) if recent_gaps else last("time/step")
    remaining = max(0, 100 - complete)
    eta = datetime.fromtimestamp(latest.wall_time + remaining * step_sec, ZoneInfo("Asia/Shanghai"))
    evals = series("eval/success_once")
    eval_text = ",".join(f"g{x.step + 1}:{x.value:.6f}" for x in evals[-5:]) or "none"
    pid = int((root / "runtime/wrapper.pid").read_text().strip())
    exit_path = root / "runtime/exit_code.txt"
    exit_code = exit_path.read_text().strip() if exit_path.exists() else "pending"
    log = (root / "runtime/driver.log").read_text(errors="replace")
    fatal = sum(log.lower().count(term) for term in (
        "traceback", "outofmemory", "cuda out of memory", "nonfinite", "workercrashed"
    ))
    checkpoints = sorted({
        int(p.name.rsplit("_", 1)[1])
        for p in root.rglob("global_step_*")
        if p.is_dir() and p.name.rsplit("_", 1)[-1].isdigit()
    })
    print(
        f"{label} alive={Path(f'/proc/{pid}').exists()} exit={exit_code} fatal={fatal} "
        f"complete={complete} success={latest.value:.6f} "
        f"mean5={mean(x.value for x in train[-5:]):.6f} "
        f"mean10={mean(x.value for x in train[-10:]):.6f} "
        f"kl={last('train/actor/approx_kl'):.6g} clip={last('train/actor/clip_fraction'):.6g} "
        f"grad={last('train/actor/grad_norm'):.6g} "
        f"tb_step_sec={last('time/step'):.1f} recent_median_sec={step_sec:.1f} "
        f"remaining={remaining} eta={eta.isoformat(timespec='minutes')} "
        f"evals=[{eval_text}] checkpoints={checkpoints[-4:]}"
    )
    if label == "prism":
        prism_tags = sorted(t for t in tags if "prism" in t.lower())
        values = " ".join(f"{tag}={last(tag):.6g}" for tag in prism_tags)
        print(f"prism_metrics {values or 'none'}")
PY

for label_run in "control:$CONTROL" "prism:$PRISM"; do
  label=${label_run%%:*}; run=${label_run#*:}
  printf '%s_progress=' "$label"
  grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:' "$run/runtime/driver.log" | tail -n 1 || true
  printf '%s_resource=' "$label"
  tail -n 1 "$run/runtime/resource.csv"
done
