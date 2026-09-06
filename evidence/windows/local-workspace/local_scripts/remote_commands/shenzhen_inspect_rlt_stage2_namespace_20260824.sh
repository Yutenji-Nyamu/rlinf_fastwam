#!/usr/bin/env bash
set -u

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ray_address=172.17.0.1:6389
rlt=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2

printf '%s\n' '--- RLT chain owner ---'
pid=$(cat "$rlt/runtime/wrapper.pid")
ps -o pid=,ppid=,pgid=,etime=,cmd= -p "$pid" || true
printf '%s\n' '--- RLT GPU actor namespaces ---'
for actor_pid in $(nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | sort -u); do
  printf 'pid=%s\n' "$actor_pid"
  RAY_ADDRESS="$ray_address" "$venv/bin/ray" list actors \
    --filter "pid=$actor_pid" --detail 2>&1 \
    | grep -E 'actor_id:|class_name:|state:|job_id:|name:|pid:|ray_namespace:' || true
done
printf '%s\n' '--- Stage2 progress and resolved output paths ---'
tr '\r' '\n' < "$rlt/stage2/runtime/driver.log" | grep 'Global Step:' | tail -n 3 || true
grep -nE 'save_path:|video_base_dir:' "$rlt/stage2/runtime/resolved.yaml" | head -n 12 || true
