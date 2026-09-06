set -u
runtime=/root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime
log="$runtime/driver.log"
date '+time=%F %T %Z'
driver=$(tr -d '[:space:]' < "$runtime/driver.pid" 2>/dev/null || true)
printf 'driver=%s alive=' "$driver"
if test -n "$driver" && kill -0 "$driver" 2>/dev/null; then printf 'yes\n'; else printf 'no\n'; fi
printf 'exit='
if test -f "$runtime/exit_code.txt"; then cat "$runtime/exit_code.txt"; else printf 'none\n'; fi
printf 'resume_and_step\n'
grep -aE 'Resuming training|Global Step:' "$log" 2>/dev/null | tail -5 || true
printf 'latest_counters\n'
grep -aE 'qam/(global_total_inserts|critic_updates|fine_updates|fine_policy_version|phase)=' "$log" 2>/dev/null | tail -12 || true
printf 'fatal\n'
grep -aEi 'CUDA out of memory|OutOfMemoryError|NCCL.*(error|abort)|RayActorError|produced non-finite|QAM_FORMAL_EXIT' "$log" 2>/dev/null | tail -20 || true
printf 'gpu\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'oom\n'
awk '/oom |oom_kill / {print}' /sys/fs/cgroup/memory.events
printf 'resource\n'
tail -1 "$runtime/resources.csv" 2>/dev/null || true
