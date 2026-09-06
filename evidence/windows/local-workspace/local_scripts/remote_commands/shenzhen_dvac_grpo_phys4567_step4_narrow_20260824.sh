#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3
LOG="$RUN/runtime/driver.log"

echo '=== Step4 optimization and DVAC ==='
tail -n 360 "$LOG" | grep -Ei 'Global Step:|success_once=|kl|clip|grad|loss|weight|ess|dvac|history|z_' | tail -n 100 || true

echo '=== GPU compute processes ==='
nvidia-smi pmon -c 1
mapfile -t pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u)
if (( ${#pids[@]} )); then
  ps -o user:16,pid,ppid,pgid,rss,etime,args -p "$(IFS=,; echo "${pids[*]}")" --no-headers || true
fi
