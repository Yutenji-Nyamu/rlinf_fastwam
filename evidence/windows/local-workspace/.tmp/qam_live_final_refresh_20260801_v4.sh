set -u
runtime=/root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime
run_root=/root/autodl-tmp/experiments/qam_formal_resume100_to380_20260801_v4/robotwin_adjust_bottle_qam_formal_resume100_to380_20260801_v4
log="$runtime/driver.log"
date '+time=%F %T %Z'
driver=$(tr -d '[:space:]' < "$runtime/driver.pid" 2>/dev/null || true)
printf 'driver=%s alive=' "$driver"
if test -n "$driver" && kill -0 "$driver" 2>/dev/null; then printf 'yes\n'; else printf 'no\n'; fi
printf 'exit='; if test -f "$runtime/exit_code.txt"; then cat "$runtime/exit_code.txt"; else printf 'none\n'; fi
grep -a 'Global Step:' "$log" 2>/dev/null | tail -1 || true
grep -aE 'qam/(global_total_inserts|critic_updates|fine_updates|fine_policy_version|pending_update_credit|phase|critic_loss|am_loss|q_mean|q_std_heads|td_target_mean|terminal_adjoint_norm)' "$log" 2>/dev/null | tail -12 || true
printf 'checkpoints\n'
for dir in "$run_root"/checkpoints/global_step_*; do
  test -d "$dir" || continue
  manifest="$dir/actor/qam_components/complete.json"
  printf '%s\t' "$(basename "$dir")"
  if test -f "$manifest"; then tr -d '\n' < "$manifest"; else printf 'missing_manifest'; fi
  printf '\n'
done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/oom |oom_kill / {print}' /sys/fs/cgroup/memory.events
