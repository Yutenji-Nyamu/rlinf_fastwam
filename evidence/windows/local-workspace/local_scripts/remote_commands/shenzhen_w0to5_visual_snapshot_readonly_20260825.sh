#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1
LOG="$RUN/runtime/driver.log"

date --iso-8601=seconds
TZ=Asia/Shanghai date --iso-8601=seconds

pid=$(cat "$RUN/runtime/wrapper.pid")
if kill -0 "$pid" 2>/dev/null; then
  echo "wrapper_alive=yes pid=$pid"
else
  echo "wrapper_alive=no pid=$pid"
fi
if [[ -f "$RUN/runtime/exit_code.txt" ]]; then
  printf 'exit_code='; cat "$RUN/runtime/exit_code.txt"
else
  echo 'exit_code=pending'
fi
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$LOG" || true

echo '-- complete steps and eval --'
grep -aE 'Global Step:|success_at_end=|success_once=' "$LOG" | tail -n 14 || true
echo '-- latest metrics --'
grep -aEi 'approx_kl=|clip_fraction=|grad_norm=|dvac_(weight|current|history|warmup|z_)' "$LOG" | tail -n 24 || true
echo '-- active rollout --'
grep -aE 'Generating Rollout Epochs:' "$LOG" | tail -n 8 || true

echo '-- files --'
stat -c '%s %y %n' "$LOG" "$RUN/runtime/resource.csv" "$RUN/resolved.yaml" 2>/dev/null || true
find "$RUN" -type f \( -name 'events.out.tfevents.*' -o -name '.complete' -o -name 'manifest.json' \) -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -k2 || true
find "$RUN" -type d -name 'global_step_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -V || true

echo '-- resource tail --'
tail -n 12 "$RUN/runtime/resource.csv" || true
echo '-- run footprint --'
du -sh "$RUN" 2>/dev/null || true

echo '-- gpu --'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits

echo SZ_W0TO5_VISUAL_SNAPSHOT_OK
