set -u
runtime=/root/autodl-tmp/experiment_exports/qam_prompt_fixcheck_20260801_v2/runtime
run=/root/autodl-tmp/experiments/qam_prompt_fixcheck_20260801_v2
date --iso-8601=seconds
printf 'EXIT='; test -f "$runtime/exit_code.txt" && cat "$runtime/exit_code.txt" || printf 'running\n'
printf 'TRAIN_PIDS='; pgrep -fc '[t]rain_embodied_agent.py' || true
printf 'RAYLETS='; pgrep -xc raylet || true
printf 'ERRORS_BEGIN\n'
grep -E 'ValueError: QAM|Traceback \(most recent call last\)|CUDA out of memory|RayActorError|NCCL' "$runtime/driver.log" 2>/dev/null | tail -n 12 || true
printf 'ERRORS_END\n'
printf 'PROGRESS_BEGIN\n'
grep -E 'global_step|fine_updates|am_loss|critic_updates|policy_version|success_once' "$runtime/driver.log" 2>/dev/null | tail -n 20 || true
find "$run" -type f -name metrics.log -exec tail -n 5 {} \; 2>/dev/null || true
printf 'PROGRESS_END\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'CGROUP_CURRENT='; cat /sys/fs/cgroup/memory.current 2>/dev/null || true
printf 'CGROUP_ANON='; awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null || true
printf 'OOM_EVENTS='; awk '$1=="oom"||$1=="oom_kill"{printf "%s=%s ",$1,$2}' /sys/fs/cgroup/memory.events 2>/dev/null || true; printf '\n'
