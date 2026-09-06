#!/usr/bin/env bash
set -euo pipefail

RLT=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix
RUNTIME="$RLT/runtime"
LOG="$RUNTIME/driver.log"
CKPT="$RLT/robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_optimizer_warmup_v3/checkpoints"

printf 'MARKER=SZ_DVAC_FORMAL_RLT_LIVE_PREFLIGHT_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
printf '%s\n' '=== RLT ==='
pid=$(cat "$RUNTIME/wrapper.pid" 2>/dev/null || true)
printf 'wrapper_pid=%s\n' "$pid"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo wrapper_alive=yes; else echo wrapper_alive=no; fi
if [[ -s "$RUNTIME/exit_code.txt" ]]; then printf 'exit_code='; cat "$RUNTIME/exit_code.txt"; else echo exit_code=pending; fi
grep -aE 'Global Step:|Traceback|OutOfMemory|WorkerCrashed|nonfinite|ERROR' "$LOG" 2>/dev/null | tail -n 18 || true
find "$CKPT" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -Vu | tail -n 12

printf '%s\n' '=== GPU AND OWNERS ==='
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u | tr '\n' ' ')
printf 'compute_pids=%s\n' "$pids"
if [[ -n "$pids" ]]; then
  ps -o user=,pid=,ppid=,etimes=,rss=,args= -p $pids || true
fi

printf '%s\n' '=== HOST AND RAY ==='
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
df -h / /home /data
RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray status | sed -n '1,34p'
printf 'MARKER=SZ_DVAC_FORMAL_RLT_LIVE_PREFLIGHT_OK\n'

