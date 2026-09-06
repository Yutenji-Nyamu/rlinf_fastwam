#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1

python=/root/autodl-tmp/RLinf/.venv/bin/python
tb_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1/tensorboard

"$python" -B - "$tb_root" <<'PY'
import json
import math
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS

acc = EventAccumulator(sys.argv[1], size_guidance={SCALARS: 0})
acc.Reload()
result = {"generated_from": sys.argv[1], "scalars": {}}
for tag in sorted(acc.Tags().get("scalars", [])):
    if not (tag.startswith("train/ogpo/") or tag.startswith("eval/")):
        continue
    values = acc.Scalars(tag)
    result["scalars"][tag] = [
        {
            "wall_time": item.wall_time,
            "step": item.step,
            "value": item.value if math.isfinite(item.value) else str(item.value),
        }
        for item in values
    ]
print(json.dumps(result, indent=2, sort_keys=True))
PY
