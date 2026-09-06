#!/usr/bin/env bash
set -u

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
LOG="$RUN/runtime/driver.log"

echo '=== time_and_run ==='
TZ=Asia/Shanghai date --iso-8601=seconds
pid=$(cat "$RUN/runtime/wrapper.pid" 2>/dev/null || true)
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "wrapper_alive=yes pid=$pid"
else
  echo "wrapper_alive=no pid=${pid:-missing}"
fi
if [[ -f "$RUN/runtime/exit_code.txt" ]]; then
  printf 'exit_code='; cat "$RUN/runtime/exit_code.txt"
else
  echo 'exit_code=pending'
fi
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$LOG" 2>/dev/null || true
grep -aE 'Global Step:|Generating Rollout Epochs:|success_once=' "$LOG" 2>/dev/null | tail -n 18 || true

echo '=== latest_checkpoints ==='
find "$RUN" -type d -name 'global_step_*' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort -V | tail -n 8 || true

echo '=== gpu ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true

echo '=== host_storage ==='
free -h
df -hT / /home /data
printf 'memory_pressure='; tr '\n' ' ' </proc/pressure/memory; echo

echo '=== public_process_rows_other_users ==='
ps -eo user:20,pid,ppid,etimes,%cpu,%mem,rss,stat,comm,args --sort=-rss \
  | awk '$1 ~ /^(liwenbo|zhangwei|guorenjie|zhuanghuiping|qiufuwen)$/ {print}' \
  | head -n 80

echo '=== public_download_command_lines ==='
pgrep -af 'hf download|huggingface-cli|Wan-AI|StarVLA|Qwen3|watchdog' 2>/dev/null \
  | grep -v 'shenzhen_dvac_grpo_v4_user_processes_and_status_narrow' \
  | head -n 60 || true

echo SZ_DVAC_GRPO_V4_NARROW_REFRESH_OK
