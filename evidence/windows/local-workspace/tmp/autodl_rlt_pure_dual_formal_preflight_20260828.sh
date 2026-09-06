#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
printf 'time=%s\nhost=%s\nuid=%s\n' "$(date -Is)" "$(hostname)" "$(id -u)"
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'memory_current='; cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp | tail -1
ps -eo pid,pgid,stat,args | grep -E 'ray::|ray start|train_embodied_agent|rlt_single_gpu' | grep -v grep || true
for p in \
  /root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1 \
  /root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1 \
  /root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_single_gpu_formal480_20260828_v1; do
  test ! -e "$p" && printf 'target_absent=%s\n' "$p" || printf 'target_exists=%s\n' "$p"
done
