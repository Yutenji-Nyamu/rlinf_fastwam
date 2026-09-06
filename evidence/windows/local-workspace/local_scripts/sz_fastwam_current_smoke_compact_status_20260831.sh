#!/usr/bin/env bash
set -u
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
NAME=fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload
RUN="$ROOT/runs/$NAME"
PACKET="$ROOT/packets/$NAME"
RELOAD="$ROOT/runs/${NAME}-reloadcheck"
pid=$(cat "$PACKET/worker.pid" 2>/dev/null || true)
if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then state=alive; else state=exited; fi
rpid=$(cat "$PACKET/reloadcheck_worker.pid" 2>/dev/null || true)
if test -n "$rpid" && kill -0 "$rpid" 2>/dev/null; then rstate=alive; else rstate=exited; fi
printf 'worker=%s pid=%s reload=%s reload_pid=%s mem_available_gib=%.1f\n' "$state" "$pid" "$rstate" "$rpid" "$(awk '/^MemAvailable:/ {print $2/1024/1024}' /proc/meminfo)"
nvidia-smi -i 2,3,4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' 'progress:'
grep -aE 'Generating Rollout|global_step|reward|success|policy_loss|grad_norm|Saving checkpoint|Training finished|Resuming training|FASTWAM' "$RUN/runtime/driver.log" "$RELOAD/runtime/driver.log" 2>/dev/null | tail -25 || true
printf '%s\n' 'resources:'
tail -3 "$RUN/runtime/resource.csv" 2>/dev/null || true
printf '%s\n' 'markers:'
for p in "$RUN/runtime/exit_code.txt" "$RELOAD/runtime/exit_code.txt" "$PACKET/SMOKE_OK"; do
  if test -e "$p"; then printf '%s=' "$(basename "$p")"; cat "$p"; else printf '%s=missing\n' "$(basename "$p")"; fi
done
printf 'checkpoint_file_count=%s\n' "$(find "$RUN/checkpoints" -type f 2>/dev/null | wc -l)"
printf '%s\n' 'reload_tail:'
tail -20 "$RELOAD/runtime/driver.log" 2>/dev/null || true
grep -aEina 'traceback|fatal|out of memory|CUDA error|NCCL error|RayTaskError|ActorDiedError|nonfinite|nan' "$RUN/runtime/driver.log" "$RELOAD/runtime/driver.log" 2>/dev/null | tail -10 || true
