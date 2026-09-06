set -u
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/runtime
printf 'RUNTIME_FILES\n'
ls -la "$runtime"
printf 'DRIVER_STATUS\n'
if [ -f "$runtime/driver_pid.txt" ]; then
  driver_pid=$(cat "$runtime/driver_pid.txt")
  ps -p "$driver_pid" -o pid=,ppid=,etimes=,%cpu=,%mem=,cmd= || true
fi
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'LOG_TAIL\n'
tail -160 "$runtime/driver.log" 2>/dev/null || true
printf 'RESOURCE_TAIL\n'
tail -8 "$runtime/resources.csv" 2>/dev/null || true
