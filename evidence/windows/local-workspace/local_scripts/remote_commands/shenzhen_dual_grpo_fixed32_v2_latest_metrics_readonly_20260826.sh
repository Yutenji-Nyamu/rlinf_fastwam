#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
DVAC=$ROOT/dvac-global-z-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v2

"$VENV/bin/python" - "$CONTROL" "$DVAC" <<'PY'
from pathlib import Path
import math, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tags = [
    "env/success_once", "env/num_trajectories",
    "train/actor/approx_kl", "train/actor/clip_fraction",
    "train/actor/grad_norm", "train/actor/lr",
    "train/actor/dvac_warmup", "train/actor/dvac_weight_mean",
    "train/actor/dvac_weight_ess_fraction", "time/step",
]
for label, root in zip(("control", "dvac"), map(Path, sys.argv[1:])):
    events = sorted((root / "tensorboard").glob("events.out.tfevents.*"))
    if not events:
        print(f"{label} no_tensorboard_event")
        continue
    ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    values = {}
    for tag in tags:
        if tag in available and ea.Scalars(tag):
            event = ea.Scalars(tag)[-1]
            values[tag] = (event.step, event.value)
            assert math.isfinite(event.value), (label, tag, event.value)
    print(label, " ".join(f"{tag}=step{step}:{value:.6g}" for tag, (step, value) in values.items()))
PY

for label_run in "control:$CONTROL" "dvac:$DVAC"; do
  label=${label_run%%:*}; run=${label_run#*:}
  printf '%s_resource_tail=' "$label"; tail -n 1 "$run/runtime/resource.csv"
done


