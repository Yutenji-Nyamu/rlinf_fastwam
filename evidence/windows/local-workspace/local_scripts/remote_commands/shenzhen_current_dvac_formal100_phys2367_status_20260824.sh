#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys2367-v2
RLT=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix
printf 'MARKER=SZ_CURRENT_DVAC_FORMAL100_PHYS2367_STATUS_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
pid=$(cat "$RUN/runtime/wrapper.pid")
if kill -0 "$pid" 2>/dev/null; then echo formal_wrapper_alive=yes; else echo formal_wrapper_alive=no; fi
if [[ -s "$RUN/runtime/exit_code.txt" ]]; then printf 'formal_exit='; cat "$RUN/runtime/exit_code.txt"; else echo formal_exit=pending; fi
grep -aE 'Global Step:|Generating Rollout|FlexiblePlacementStrategy|Traceback|OutOfMemory|WorkerCrashed|nonfinite|ERROR' "$RUN/runtime/driver.log" | tail -n 40 || true
printf '%s\n' '=== GPU ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' '=== RLT LAST ==='
grep -aE 'Global Step:|Traceback|OutOfMemory|WorkerCrashed|nonfinite|ERROR' "$RLT/runtime/driver.log" | tail -n 8 || true
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo
printf 'MARKER=SZ_CURRENT_DVAC_FORMAL100_PHYS2367_STATUS_OK\n'
