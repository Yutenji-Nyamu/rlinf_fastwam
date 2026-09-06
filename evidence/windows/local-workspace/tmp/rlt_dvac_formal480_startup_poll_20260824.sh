set -euo pipefail
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
run_root=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
printf 'LIFECYCLE\n'
for f in driver_pid.txt monitor_pid.txt started_at.txt exit_code.txt finished_at.txt source_head.txt; do
  if [ -f "$runtime/$f" ]; then printf '%s=' "$f"; cat "$runtime/$f"; fi
done
printf 'PROCESSES\n'
pgrep -af 'train_embodied_agent.py|ray::|raylet|gcs_server|resource_monitor.sh' || true
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'CGROUP\n'
printf 'current='; cat /sys/fs/cgroup/memory.current
printf 'high='; cat /sys/fs/cgroup/memory.high
printf 'max='; cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
printf 'RESOURCE_TAIL\n'
tail -n 5 "$runtime/resources.csv" || true
printf 'FOREGROUND_TAIL\n'
tail -n 100 "$runtime/foreground.log" || true
printf 'DRIVER_TAIL\n'
tail -n 80 "$runtime/driver.log" 2>/dev/null || true
printf 'METRICS_TAIL\n'
tail -n 80 "$run_root/metrics.log" 2>/dev/null || true
