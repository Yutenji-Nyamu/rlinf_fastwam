#!/usr/bin/env bash
set -u

DVAC_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3
RLT_RUN=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix
RLT_EXP="$RLT_RUN/robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_optimizer_warmup_v3"

echo '=== DVAC 4567 brief health ==='
if [[ -f "$DVAC_RUN/runtime/wrapper.pid" ]]; then
  pid=$(<"$DVAC_RUN/runtime/wrapper.pid")
  if kill -0 "$pid" 2>/dev/null; then echo "wrapper_alive=yes pid=$pid"; else echo "wrapper_alive=no pid=$pid"; fi
fi
if [[ -f "$DVAC_RUN/runtime/exit_code.txt" ]]; then
  echo -n 'exit_code='; cat "$DVAC_RUN/runtime/exit_code.txt"
else
  echo 'exit_code=pending'
fi
if [[ -f "$DVAC_RUN/runtime/driver.log" ]]; then
  grep -E 'Global Step|Generating Rollout|Train Epoch|Traceback|OutOfMemory|WorkerCrashed|CUDA out of memory' "$DVAC_RUN/runtime/driver.log" | tail -n 20 || true
fi
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '5,8p'

echo '=== RLT final small-file inventory ==='
printf 'exit_code='; cat "$RLT_RUN/runtime/exit_code.txt" 2>/dev/null || echo missing
printf 'started_at='; cat "$RLT_RUN/runtime/started_at.txt" 2>/dev/null || echo missing
printf 'finished_at='; cat "$RLT_RUN/runtime/finished_at.txt" 2>/dev/null || echo missing
find "$RLT_RUN" -maxdepth 3 -type f -size -20M -printf '%s\t%p\n' | sort -n
echo '=== RLT checkpoint250 inventory ==='
find "$RLT_EXP/checkpoints/global_step_250" -maxdepth 3 -type f -printf '%s\t%p\n' 2>/dev/null | sort -n | tail -n 80
