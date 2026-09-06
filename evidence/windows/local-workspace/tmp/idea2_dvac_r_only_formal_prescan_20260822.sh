#!/usr/bin/env bash
set -u

source_root=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822

hostname
pwd
id -u
date --iso-8601=seconds
git -C "$source_root" status --short --branch
git -C "$source_root" rev-parse HEAD
git -C "$source_root" rev-parse --abbrev-ref HEAD
git -C "$source_root" remote -v
ps -eo pid,ppid,pgid,stat,cmd --sort=pid | grep -E 'idea2_dvac|train_embodied_agent|ray::|raylet' | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
if [[ -e "$run_dir" ]]; then printf 'RUN_TARGET_EXISTS\n'; else printf 'RUN_TARGET_ABSENT\n'; fi
if [[ -e "$runtime_dir" ]]; then printf 'RUNTIME_TARGET_EXISTS\n'; else printf 'RUNTIME_TARGET_ABSENT\n'; fi
