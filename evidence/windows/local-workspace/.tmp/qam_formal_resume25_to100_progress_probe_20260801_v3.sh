set -u
runtime=/root/autodl-tmp/experiment_exports/qam_formal_resume25_to100_20260801_v3/runtime
log="$runtime/driver.log"
date '+time=%F %T %Z'
printf 'alive='
driver=$(tr -d '[:space:]' < "$runtime/driver.pid" 2>/dev/null || true)
if test -n "$driver" && kill -0 "$driver" 2>/dev/null; then printf 'yes\n'; else printf 'no\n'; fi
printf 'exit='
if test -f "$runtime/exit_code.txt"; then cat "$runtime/exit_code.txt"; else printf 'none\n'; fi
printf 'progress\n'
grep -aEi 'resume|restor|checkpoint|global_total_inserts|critic_updates|fine_updates|policy_version|qam/|step[^a-z]|fatal|traceback|error' "$log" 2>/dev/null | tail -100 || true
printf 'resource\n'
tail -1 "$runtime/resources.csv" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
awk '/oom |oom_kill / {print}' /sys/fs/cgroup/memory.events 2>/dev/null || true
