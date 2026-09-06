#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
run=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
printf 'RLT_BRANCH=%s\n' "$(git -C "$repo" branch --show-current)"
printf 'RLT_HEAD=%s\n' "$(git -C "$repo" rev-parse HEAD)"
printf 'RLT_UPSTREAM=%s\n' "$(git -C "$repo" rev-parse --abbrev-ref '@{upstream}')"
printf 'RLT_AHEAD_BEHIND='; git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
printf 'RLT_STATUS\n'; git -C "$repo" status --short --branch
printf 'STALE_RLT_GIT\n'
ps -eo pid=,comm=,args= | awk '$2 == "git" && $0 ~ /RLinf_rlt_teacher_dvac/ {print}'

printf 'GLOBAL_Z_PROCESSES\n'
for name in wrapper driver observer; do
  pid=$(cat "$run/$name.pid" 2>/dev/null || true)
  printf '%s=%s alive=%s\n' "$name" "$pid" "$([ -n "$pid" ] && [ -d "/proc/$pid" ] && echo 1 || echo 0)"
done
grep -a 'Global Step' "$run/driver.log" 2>/dev/null | tail -n 1 || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
grep -E '^oom |^oom_kill ' /sys/fs/cgroup/memory.events
