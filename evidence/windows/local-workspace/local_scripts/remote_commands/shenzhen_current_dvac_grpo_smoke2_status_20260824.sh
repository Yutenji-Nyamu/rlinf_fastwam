#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-smoke2-4gpu128train64eval-v1
RUNTIME="$RUN/runtime"
LOG="$RUNTIME/driver.log"

printf 'MARKER=SZ_CURRENT_DVAC_GRPO_SMOKE2_STATUS_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
pid=$(cat "$RUNTIME/wrapper.pid")
if kill -0 "$pid" 2>/dev/null; then printf 'wrapper_alive=yes\n'; else printf 'wrapper_alive=no\n'; fi
if [[ -s "$RUNTIME/exit_code.txt" ]]; then printf 'exit_code='; cat "$RUNTIME/exit_code.txt"; else printf 'exit_code=pending\n'; fi

printf '%s\n' '=== PROGRESS ==='
grep -aE 'Global Step:|dvac/|weight|success|reward=|eval=|Traceback|OutOfMemory|WorkerCrashed|nonfinite|ERROR' "$LOG" 2>/dev/null | tail -n 70 || true
printf '%s\n' '=== DRIVER TAIL ==='
tail -n 35 "$LOG" 2>/dev/null || true

printf '%s\n' '=== ARTIFACTS ==='
find "$RUN" -maxdepth 8 -type d -name 'global_step_*' -printf '%p\n' 2>/dev/null | sort -V | tail -n 6
find "$RUN" -type f \( -iname '*dvac*' -o -iname '*weight*' \) -printf '%s %p\n' 2>/dev/null | sort -n | tail -n 20

printf '%s\n' '=== RESOURCE TAIL ==='
tail -n 8 "$RUNTIME/resource.csv" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
grep -E '^(MemAvailable|SwapFree):' /proc/meminfo
printf 'MARKER=SZ_CURRENT_DVAC_GRPO_SMOKE2_STATUS_OK\n'
