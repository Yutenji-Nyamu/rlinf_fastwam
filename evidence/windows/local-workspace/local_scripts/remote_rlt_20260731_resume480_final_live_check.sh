#!/usr/bin/env bash
set -euo pipefail

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_formal_resume250_to480_20260730_v1/runtime
checkpoint=/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1/robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1/checkpoints/global_step_480

printf 'now=%s\n' "$(date --iso-8601=seconds)"
printf 'exit_code=%s\n' "$(cat "$runtime/exit_code.txt")"
printf 'finished_at=%s\n' "$(cat "$runtime/finished_at.txt")"
for name in driver monitor; do
  pid="$(cat "$runtime/${name}_pid.txt")"
  if kill -0 "$pid" 2>/dev/null; then
    printf '%s_pid=%s alive=1\n' "$name" "$pid"
  else
    printf '%s_pid=%s alive=0\n' "$name" "$pid"
  fi
done
printf 'train_processes=%s\n' "$(
  ps -eo comm=,args= |
    awk '$1 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {count++} END {print count+0}'
)"
printf 'raylet_processes=%s\n' "$(pgrep -xc raylet || true)"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'cgroup_current=%s\n' "$(cat /sys/fs/cgroup/memory.current)"
printf 'cgroup_anon=%s\n' "$(awk '$1 == "anon" {print $2}' /sys/fs/cgroup/memory.stat)"
printf 'cgroup_file=%s\n' "$(awk '$1 == "file" {print $2}' /sys/fs/cgroup/memory.stat)"
printf 'memory_events=%s\n' "$(tr '\n' ' ' </sys/fs/cgroup/memory.events)"
printf 'disk_available=%s\n' "$(
  df -B1 --output=avail /root/autodl-tmp | tail -n 1 | tr -d ' '
)"
python - "$checkpoint/actor/sac_components/rlt_trainer_state/rlt_trainer_state_complete.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(
    "final_checkpoint="
    f"complete:{payload['complete']},"
    f"step:{payload['saved_runner_step']},"
    f"update_step:{payload['update_step']},"
    f"world_size:{payload['actor_world_size']}"
)
PY
