set -u
runtime=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1/runtime
printf 'OWNED_PIDS\n'
for pid_file in driver_pid.txt monitor_pid.txt; do
  pid=$(cat "$runtime/$pid_file" 2>/dev/null || true)
  printf '%s=%s ' "$pid_file" "$pid"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo alive; else echo exited; fi
done
printf 'MATCHED_PROCESSES\n'
ps -eo pid,ppid,pgid,etimes,cmd | grep -E 'rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1|RLTACFSDPPolicy|MultiStepRolloutWorker|EnvWorker' | grep -v grep || true
printf 'GPU\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
printf 'CGROUP_EVENTS\n'
cat /sys/fs/cgroup/memory.events
