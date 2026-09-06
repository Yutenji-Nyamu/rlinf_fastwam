set -u
runtime=/root/autodl-tmp/experiment_exports/qam_formal_resume25_to100_20260801_v3/runtime
run_root=/root/autodl-tmp/experiments/qam_formal_resume25_to100_20260801_v3/robotwin_adjust_bottle_qam_formal_resume25_to100_20260801_v3
log="$runtime/driver.log"
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin

hostname
pwd
id -u
date '+time=%F %T %Z'
printf 'git\n'
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
printf 'processes\n'
ps -eo pid,ppid,pgid,stat,lstart,cmd | grep -E 'qam_formal_resume25_to100_20260801_v3|train_embodied_agent.py|raylet' | grep -v grep || true
printf 'exit_files\n'
for file in "$runtime/exit_code.txt" "$runtime/monitor_exit_code.txt"; do
  if test -f "$file"; then
    printf '%s=%s\n' "$(basename "$file")" "$(tr -d '[:space:]' < "$file")"
  fi
done
printf 'latest_steps\n'
grep -a 'Global Step:' "$log" 2>/dev/null | tail -5 || true
printf 'latest_qam_metrics\n'
grep -aE 'qam/(global_total_inserts|critic_updates|fine_updates|am_updates_run|fine_policy_version|phase|critic_loss|am_loss|terminal_adjoint_norm|fine_grad_norm|prefix_roundtrip|max_q|q_)' "$log" 2>/dev/null | tail -80 || true
printf 'latest_success_metrics\n'
grep -aE '(success|reward|episode)' "$log" 2>/dev/null | tail -40 || true
printf 'fatal_since_resume\n'
grep -aEi 'CUDA out of memory|OutOfMemoryError|NCCL.*(error|abort)|RayActorError|nan|inf|QAM frozen-prefix.*produced non-finite|QAM frozen-prefix.*changed|QAM_FORMAL_EXIT' "$log" 2>/dev/null | tail -50 || true
printf 'checkpoints\n'
for dir in "$run_root"/checkpoints/global_step_*; do
  test -d "$dir" || continue
  size=$(du -sb "$dir" | awk '{print $1}')
  manifest="$dir/actor/qam_components/complete.json"
  printf '%s\t%s\t' "$(basename "$dir")" "$size"
  if test -f "$manifest"; then tr -d '\n' < "$manifest"; else printf 'missing_manifest'; fi
  printf '\n'
done
printf 'resources_header_tail\n'
head -2 "$runtime/resources.csv" 2>/dev/null || true
tail -5 "$runtime/resources.csv" 2>/dev/null || true
printf 'gpu\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits
printf 'cgroup\n'
printf 'memory.current='; cat /sys/fs/cgroup/memory.current 2>/dev/null || true
awk '/^(anon|file) / {print}' /sys/fs/cgroup/memory.stat 2>/dev/null || true
cat /sys/fs/cgroup/memory.events 2>/dev/null || true
printf 'disk\n'
df -B1 /root/autodl-tmp | tail -1
