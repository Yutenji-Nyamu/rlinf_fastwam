#!/usr/bin/env bash
set -u

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ray_address=172.17.0.1:6389

printf '%s\n' '--- exact old DSRL v1 GPU actor PIDs ---'
for pid in 344717 344719 344723 344725 344727 344782; do
  if kill -0 "$pid" 2>/dev/null; then
    ps -o pid=,ppid=,pgid=,etime=,cmd= -p "$pid"
  else
    printf '%s state=dead\n' "$pid"
  fi
done
printf '%s\n' '--- Ray actors in DSRL v1 namespace ---'
RAY_ADDRESS="$ray_address" "$venv/bin/ray" list actors \
  --filter 'ray_namespace=RLinf_1' --detail 2>&1 | sed -n '1,300p' || true
