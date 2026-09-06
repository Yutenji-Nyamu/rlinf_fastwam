#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
LOG=$RUN/runtime/driver.log

date '+%F %T %Z'
pid=$(<"$RUN/runtime/wrapper.pid")
if kill -0 "$pid" 2>/dev/null; then echo "wrapper_alive=yes pid=$pid"; else echo "wrapper_alive=no pid=$pid"; fi
if [[ -f "$RUN/runtime/exit_code.txt" ]]; then printf 'exit_code='; cat "$RUN/runtime/exit_code.txt"; else echo 'exit_code=pending'; fi
grep -E 'Global Step:|Generating Rollout Epochs:|Actor initialized|Env initialized|Rollout initialized|success_once=' "$LOG" | tail -n 20 || true
printf 'fatal_matches='; grep -Eci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$LOG" || true
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,2p'

