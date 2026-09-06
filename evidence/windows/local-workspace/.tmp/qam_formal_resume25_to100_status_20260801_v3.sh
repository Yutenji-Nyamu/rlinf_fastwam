set -u
runtime=/root/autodl-tmp/experiment_exports/qam_formal_resume25_to100_20260801_v3/runtime
run_root=/root/autodl-tmp/experiments/qam_formal_resume25_to100_20260801_v3/robotwin_adjust_bottle_qam_formal_resume25_to100_20260801_v3

date '+time=%F %T %Z'
printf 'processes\n'
ps -eo pid,ppid,pgid,stat,lstart,cmd | grep -E 'qam_formal_resume25_to100_20260801_v3|train_embodied_agent.py|raylet' | grep -v grep || true
printf 'exit\n'
for file in "$runtime/exit_code.txt" "$runtime/monitor_exit_code.txt"; do
  if test -f "$file"; then
    printf '%s=%s\n' "$(basename "$file")" "$(tr -d '[:space:]' < "$file")"
  fi
done
printf 'driver_tail\n'
tail -120 "$runtime/driver.log" 2>/dev/null || true
printf 'resources_tail\n'
tail -5 "$runtime/resources.csv" 2>/dev/null || true
printf 'gpu\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'oom\n'
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
printf 'checkpoints\n'
find "$run_root/checkpoints" -maxdepth 2 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sort -V || true
