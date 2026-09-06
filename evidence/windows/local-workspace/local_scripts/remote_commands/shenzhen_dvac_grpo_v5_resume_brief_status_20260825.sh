#!/usr/bin/env bash
set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30
LOG="$RUN/runtime/driver.log"
TZ=Asia/Shanghai date --iso-8601=seconds
pid=$(cat "$RUN/runtime/wrapper.pid" 2>/dev/null || true)
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo "wrapper_alive=yes pid=$pid"; else echo "wrapper_alive=no pid=${pid:-missing}"; fi
if [[ -f "$RUN/runtime/exit_code.txt" ]]; then printf 'exit_code='; cat "$RUN/runtime/exit_code.txt"; else echo 'exit_code=pending'; fi
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$LOG" 2>/dev/null || true
grep -aE 'Resuming training|global_step_30|DVAC|Global Step:|Generating Rollout Epochs:' "$LOG" 2>/dev/null | tail -n 30 || true
echo '-- gpu4567 --'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 4,5,6,7 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
free -h | sed -n '1,3p'
echo '-- driver tail --'
tail -n 35 "$LOG" 2>/dev/null || true
echo SZ_DVAC_GRPO_V5_RESUME_BRIEF_STATUS_OK
