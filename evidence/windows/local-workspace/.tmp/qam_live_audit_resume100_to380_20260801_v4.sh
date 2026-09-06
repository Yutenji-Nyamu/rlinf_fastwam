set -u

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
runtime=/root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime
run_root=/root/autodl-tmp/experiments/qam_formal_resume100_to380_20260801_v4/robotwin_adjust_bottle_qam_formal_resume100_to380_20260801_v4
log="$runtime/driver.log"
v2=/root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime/driver.log
v3=/root/autodl-tmp/experiment_exports/qam_formal_resume25_to100_20260801_v3/runtime/driver.log

hostname
pwd
id -u
date '+time=%F %T %Z'
printf 'git\n'
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
printf 'driver\n'
driver=$(tr -d '[:space:]' < "$runtime/driver.pid" 2>/dev/null || true)
printf 'pid=%s alive=' "$driver"
if test -n "$driver" && kill -0 "$driver" 2>/dev/null; then printf 'yes\n'; else printf 'no\n'; fi
printf 'training_processes='
pgrep -af '[t]rain_embodied_agent.py|[r]aylet|[g]cs_server' | wc -l
printf 'exit\n'
for file in "$runtime/exit_code.txt" "$runtime/monitor_exit_code.txt"; do
  if test -f "$file"; then printf '%s=%s\n' "$(basename "$file")" "$(tr -d '[:space:]' < "$file")"; fi
done
printf 'latest_steps\n'
grep -a 'Global Step:' "$log" 2>/dev/null | tail -5 || true
printf 'latest_metrics\n'
grep -aE 'qam/(global_total_inserts|critic_updates|fine_updates|fine_policy_version|pending_update_credit|phase|critic_loss|critic_grad_norm|q_mean|q_std_heads|td_target_mean|am_loss|terminal_adjoint_norm|fine_grad_norm|prefix_roundtrip)' "$log" 2>/dev/null | tail -50 || true
printf 'success_all\n'
{ grep -aoE 'success_once=[0-9.]+' "$v2" 2>/dev/null | head -25; grep -aoE 'success_once=[0-9.]+' "$v3" 2>/dev/null; grep -aoE 'success_once=[0-9.]+' "$log" 2>/dev/null; } | cut -d= -f2 | awk '{n++;s+=$1} END {printf "cycles=%d episodes=%d successes=%.0f rate=%.6f\n",n,2*n,2*s,(n?s/n:0)}'
for window in 10 20 50; do
  printf 'success_last_%s=' "$window"
  { grep -aoE 'success_once=[0-9.]+' "$v2" 2>/dev/null | head -25; grep -aoE 'success_once=[0-9.]+' "$v3" 2>/dev/null; grep -aoE 'success_once=[0-9.]+' "$log" 2>/dev/null; } | cut -d= -f2 | tail -n "$window" | awk '{n++;s+=$1} END {printf "cycles=%d episodes=%d successes=%.0f rate=%.6f\n",n,2*n,2*s,(n?s/n:0)}'
done
printf 'fatal\n'
grep -aEi 'CUDA out of memory|OutOfMemoryError|NCCL.*(error|abort)|RayActorError|produced non-finite|Traceback \(most recent call last\)|QAM_FORMAL_EXIT' "$log" 2>/dev/null | tail -20 || true
printf 'checkpoints\n'
for dir in "$run_root"/checkpoints/global_step_*; do
  test -d "$dir" || continue
  manifest="$dir/actor/qam_components/complete.json"
  printf '%s\t%s\t' "$(basename "$dir")" "$(du -sb "$dir" | awk '{print $1}')"
  if test -f "$manifest"; then tr -d '\n' < "$manifest"; else printf 'missing_manifest'; fi
  printf '\n'
done
printf 'files\n'
wc -c "$log" "$runtime/resources.csv" 2>/dev/null || true
find "$run_root" -type f -name 'events.out.tfevents*' -printf '%s\t%p\n' 2>/dev/null | sort -n || true
printf 'resource_tail\n'
head -1 "$runtime/resources.csv" 2>/dev/null || true
tail -3 "$runtime/resources.csv" 2>/dev/null || true
printf 'gpu\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits
printf 'cgroup\n'
printf 'memory.current='; cat /sys/fs/cgroup/memory.current 2>/dev/null || true
awk '/^(anon|file) / {print}' /sys/fs/cgroup/memory.stat 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
printf 'disk\n'
df -B1 /root/autodl-tmp | tail -1
