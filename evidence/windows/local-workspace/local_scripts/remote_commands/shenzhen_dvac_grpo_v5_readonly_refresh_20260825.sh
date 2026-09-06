#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30

TZ=Asia/Shanghai date --iso-8601=seconds
pid=$(cat "$RUN/runtime/wrapper.pid")
if kill -0 "$pid" 2>/dev/null; then
  echo "wrapper_alive=yes pid=$pid"
else
  echo "wrapper_alive=no pid=$pid"
fi
printf 'exit_code='; cat "$RUN/runtime/exit.code" 2>/dev/null || echo pending
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$RUN/runtime/driver.log" || true
grep -aE 'Global Step:|Generating Rollout Epochs:' "$RUN/runtime/driver.log" | tail -n 12 || true
tail -n 2 "$RUN/runtime/resource.csv"
find "$RUN" -type d -name 'global_step_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' | sort -V
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,3p'
