#!/usr/bin/env bash
set -u

ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
NAME=fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload
RUN="$ROOT/runs/$NAME"
PACKET="$ROOT/packets/$NAME"
RELOAD="$ROOT/runs/${NAME}-reloadcheck"

pid=$(cat "$PACKET/worker.pid" 2>/dev/null || true)
if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then state=alive; else state=exited; fi
printf 'worker=%s pid=%s\n' "$state" "$pid"
printf '%s\n' 'gpu_status:'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'mem_available_gib=%.1f\n' "$(awk '/^MemAvailable:/ {print $2/1024/1024}' /proc/meminfo)"
printf '%s\n' 'driver_tail:'
tail -30 "$RUN/runtime/driver.log" 2>/dev/null || true
printf '%s\n' 'resource_tail:'
tail -5 "$RUN/runtime/resource.csv" 2>/dev/null || true
printf '%s\n' 'checkpoint_files:'
find "$RUN/checkpoints" -maxdepth 5 -type f -printf '%s %p\n' 2>/dev/null | sort -n | tail -20
printf '%s\n' 'reload_tail:'
tail -15 "$RELOAD/runtime/driver.log" 2>/dev/null || true
printf '%s\n' 'terminal_markers:'
for p in "$RUN/runtime/exit_code.txt" "$RELOAD/runtime/exit_code.txt" "$PACKET/SMOKE_OK"; do
  if test -e "$p"; then printf '%s=' "$p"; cat "$p"; else printf '%s=missing\n' "$p"; fi
done
printf '%s\n' 'fatal_scan:'
grep -Eina 'traceback|fatal|out of memory|CUDA error|NCCL error|RayTaskError|ActorDiedError|nonfinite|nan' "$RUN/runtime/driver.log" "$RELOAD/runtime/driver.log" 2>/dev/null | tail -20 || true
