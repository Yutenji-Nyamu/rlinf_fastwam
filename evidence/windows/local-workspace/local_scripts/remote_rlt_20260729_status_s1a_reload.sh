set -u

RUNTIME_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_smoke_20260729_v1/s1a_reload_runtime

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
printf '%s\n' 'LOG_TAIL'
tail -n 45 "$RUNTIME_ROOT/driver.log" 2>/dev/null || true
printf '%s\n' 'RESOURCE_ROWS'
wc -l "$RUNTIME_ROOT/resources.csv" 2>/dev/null || true
tail -n 3 "$RUNTIME_ROOT/resources.csv" 2>/dev/null || true
