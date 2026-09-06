#!/usr/bin/env bash
set -u

OLD_RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821
OLD_RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_formal_100step_2gpu16env_20260821
NEW_ROOT=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight

hostname
pwd
id -u
date --iso-8601=seconds
ps -p 114146,114149,114150 -o pid=,stat=,etimes=,cmd=
grep -o 'Global Step:[[:space:]]*[0-9]\+/100' "$OLD_RUN/metrics.log" | tail -n 3
test -f "$OLD_RUNTIME/driver.exitcode" && cat "$OLD_RUNTIME/driver.exitcode"
test -f "$OLD_RUNTIME/launch_finished_at.txt" && cat "$OLD_RUNTIME/launch_finished_at.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -BG --output=avail /root/autodl-tmp | tail -n 1
git -C "$NEW_ROOT" rev-parse HEAD
git -C "$NEW_ROOT" status --short --branch
