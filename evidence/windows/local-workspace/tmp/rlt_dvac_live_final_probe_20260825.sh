#!/usr/bin/env bash
set -uo pipefail

run=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime

echo "timestamp=$(date --iso-8601=seconds)"
grep 'Global Step:' "$run/metrics.log" | tail -n 1
pgrep -af 'run_formal_rlt_teacher_dvac_fresh480_20260824|main_ppo.py|ray::ActorWorker|ray::RolloutWorker|ray::EnvWorker' || true

for name in wrapper driver monitor; do
  pid_file="$runtime/$name.pid"
  if [[ -f "$pid_file" ]]; then
    pid=$(tr -d '[:space:]' < "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name=$pid alive"
    else
      echo "$name=$pid dead"
    fi
  fi
done

printf 'cgroup_current_bytes='
cat /sys/fs/cgroup/memory.current
grep -E '^(high|max|oom|oom_kill) ' /sys/fs/cgroup/memory.events
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits
df -h /root/autodl-tmp | tail -n 1
