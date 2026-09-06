#!/usr/bin/env bash
set -u

OLD_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys2367-v2
NEW_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389

echo '=== identity/time ==='
date '+%F %T %Z'
id

echo '=== old/new wrappers ==='
for run in "$OLD_RUN" "$NEW_RUN"; do
  echo "run=$run"
  for name in wrapper.pid owned.pgid observer.pid exit_code.txt; do
    path="$run/runtime/$name"
    if [[ -f "$path" ]]; then printf '%s=' "$name"; cat "$path"; else echo "$name=missing"; fi
  done
  if [[ -f "$run/runtime/wrapper.pid" ]]; then
    pid=$(<"$run/runtime/wrapper.pid")
    ps -o user=,pid=,ppid=,pgid=,lstart=,etime=,rss=,stat=,args= -p "$pid" 2>/dev/null || true
  fi
done

echo '=== GPU summary and compute apps ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits

echo '=== exact GPU process identity ==='
mapfile -t pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
for pid in "${pids[@]}"; do
  echo "--- pid=$pid ---"
  ps -o user=,uid=,pid=,ppid=,pgid=,lstart=,etime=,rss=,stat=,args= -p "$pid" 2>/dev/null || true
  if [[ -r "/proc/$pid/cmdline" ]]; then tr '\0' ' ' < "/proc/$pid/cmdline"; echo; fi
  if [[ -r "/proc/$pid/environ" ]]; then
    tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(CUDA_VISIBLE_DEVICES|RAY_ADDRESS|RAY_JOB_ID|RAY_NAMESPACE|RLINF_CODE_WORKING_DIR)=' || true
  fi
  readlink -f "/proc/$pid/cwd" 2>/dev/null | sed 's/^/cwd=/' || true
done

echo '=== alive Ray actor records ==='
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" list actors --filter 'state=ALIVE' --detail 2>&1 | sed -n '1,900p' || true

echo '=== Ray jobs ==='
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" list jobs --detail 2>&1 | sed -n '1,600p' || true

echo '=== memory ==='
free -h | sed -n '1,2p'
grep -E 'MemAvailable|SwapFree' /proc/meminfo

