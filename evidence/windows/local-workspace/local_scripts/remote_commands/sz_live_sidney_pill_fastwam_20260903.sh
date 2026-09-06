#!/usr/bin/env bash
set -euo pipefail

SIDNEY='/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1'
FAST='/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2'

printf 'TIME=%s\n' "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"

inspect_run() {
  label="$1"
  run="$2"
  log="$run/runtime/driver.log"
  pid_file="$run/runtime/wrapper.pid"
  printf '\n===%s===\nRUN=%s\n' "$label" "$run"
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    printf 'WRAPPER_PID=%s ALIVE=' "$pid"
    if kill -0 "$pid" 2>/dev/null; then echo yes; else echo no; fi
    ps -o pid=,ppid=,etimes=,rss=,stat=,cmd= -p "$pid" || true
  else
    echo 'WRAPPER_PID=missing ALIVE=no'
  fi
  printf 'EXIT_CODE='; cat "$run/runtime/exit_code.txt" 2>/dev/null || echo pending
  [ -f "$log" ] || { echo DRIVER_LOG=missing; return; }
  printf 'LOG_BYTES='; stat -c '%s' "$log"
  echo 'LATEST_STEPS:'
  grep -a 'Global Step:' "$log" | tail -n 5 || true
  echo 'LATEST_ROLLOUT:'
  grep -aE 'Generating Rollout Epochs:|Evaluating Rollout Epochs:|Saving checkpoint|save_checkpoint' "$log" | tail -n 5 || true
  echo 'LATEST_SUCCESS:'
  grep -aoE 'success_once=[0-9]+([.][0-9]+)?' "$log" | tail -n 10 || true
  echo 'LATEST_EVAL:'
  grep -aiE 'fixed.?32|eval.*success|success.*eval|evaluation.*(success|step)|Evaluating Rollout Epochs: 100%' "$log" | tail -n 12 || true
  echo 'FATAL_COUNTS:'
  for pat in 'CUDA out of memory' 'OIDN Error' 'pthread_key_create' 'PyGILState_Release' 'ErrorInitializationFailed' 'ActorDiedError' 'WorkerCrashedError' 'Traceback' 'Fatal Python error'; do
    printf '%s=' "$pat"; grep -aic "$pat" "$log" 2>/dev/null || true
  done
  echo 'CHECKPOINTS:'
  find "$run" -type d -name 'global_step_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 4 | cut -d' ' -f2- || true
}

inspect_run SIDNEY_PILL "$SIDNEY"
inspect_run FASTWAM "$FAST"

echo '\n===GPU4_7==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits | sed -n '5,8p'

echo '\n===OWNED_GPU_PROCESSES==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true

echo '\n===RAM==='
free -b | awk '/Mem:/ {printf "total_bytes=%s used_bytes=%s available_bytes=%s\n",$2,$3,$7}'
