#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-smoke1-2gpu64x4-b1024-noeval-phys23-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

printf '%s\n' '--- terminal ---'
printf 'exit_code='; cat "$RUN/runtime/exit_code.txt"
printf 'started_at='; cat "$RUN/runtime/started_at.txt"
printf 'finished_at='; cat "$RUN/runtime/finished_at.txt"
printf 'fatal_lines='
grep -aicE 'Traceback|OutOfMemory|CUDA error|non[- ]?finite|worker.*died|RayActorError' "$RUN/runtime/driver.log" || true

printf '%s\n' '--- checkpoint candidates ---'
find "$RUN" -type f \( -path '*global_step_1*' -o -iname '*manifest*' -o -name '.metadata' \) \
  -printf '%p %s\n' | sort

printf '%s\n' '--- tensorboard scalars ---'
"$VENV/bin/python" - "$RUN" <<'PY'
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(sys.argv[1])
events = sorted(root.rglob('events.out.tfevents.*'), key=lambda p: p.stat().st_mtime)
print('event_files=', len(events))
if not events:
    raise SystemExit(0)
ea = EventAccumulator(str(events[-1]), size_guidance={'scalars': 0})
ea.Reload()
wanted = (
    'success_once', 'num_trajectories', 'advantages_', 'prism_dvac/',
    'approx_kl', 'clip_fraction', 'grad_norm', 'policy_loss', 'total_loss',
)
for tag in sorted(ea.Tags().get('scalars', [])):
    if any(key in tag for key in wanted):
        values = ea.Scalars(tag)
        if values:
            item = values[-1]
            print(f'{tag} step={item.step} value={item.value:.9g}')
PY

printf '%s\n' '--- resources ---'
awk -F, 'NR==2 {min=$3; max2=$4; max3=$6} NR>1 {if($3 && $3<min)min=$3; if($4>max2)max2=$4; if($6>max3)max3=$6} END {printf "min_mem_available_gib=%.3f max_gpu2_used_mib=%d max_gpu3_used_mib=%d\n", min/1024/1024,max2,max3}' "$RUN/runtime/resource.csv"
nvidia-smi -i 2,3,4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
