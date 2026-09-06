#!/usr/bin/env bash
set -euo pipefail
venv=/root/autodl-tmp/RLinf/.venv
run_root=/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1
event="$(
  find "$run_root/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' \
    | head -n 1
)"
"${venv}/bin/python" -B - "$event" <<'PY'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

acc = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
acc.Reload()
for tag in acc.Tags()["scalars"]:
    if not any(token in tag for token in ("rlt/", "actor_switch", "bc_weight", "q_weight")):
        continue
    values = acc.Scalars(tag)
    print(f"{tag}\t{values[-1].step}\t{values[-1].value}")
PY
