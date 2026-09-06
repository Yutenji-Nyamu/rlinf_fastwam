set -u

EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_smoke_20260729_v1
RUNTIME_ROOT="$EVIDENCE_ROOT/s1a_runtime"
RUN_ROOT=/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1
S1A_NAME=robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1

date -Is
driver_pid=$(cat "$RUNTIME_ROOT/driver_pid.txt" 2>/dev/null || printf '%s' -1)
if kill -0 "$driver_pid" 2>/dev/null; then
  printf 'STATE=RUNNING\nDRIVER_PID=%s\n' "$driver_pid"
else
  printf 'STATE=FINISHED\nDRIVER_PID=%s\n' "$driver_pid"
fi
if test -f "$RUNTIME_ROOT/exit_code.txt"; then
  printf 'EXIT_CODE=%s\n' "$(cat "$RUNTIME_ROOT/exit_code.txt")"
fi

printf '%s\n' 'GPU_NOW'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
printf '%s\n' 'RAM_NOW'
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo
printf '%s\n' 'LOG_TAIL'
tail -n 35 "$RUNTIME_ROOT/driver.log" 2>/dev/null || true
printf '%s\n' 'RESOURCE_ROWS'
wc -l "$RUNTIME_ROOT/resources.csv" 2>/dev/null || true
tail -n 3 "$RUNTIME_ROOT/resources.csv" 2>/dev/null || true
printf '%s\n' 'CHECKPOINT_FILES'
find "$RUN_ROOT/s1a/$S1A_NAME" -maxdepth 5 -type f \
  -printf '%p\t%s\n' 2>/dev/null | sort | tail -n 30
