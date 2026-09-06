#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = \
  codex/ogpo-pi0-robotwin
printf 'STATUS_ALL\n'
git -C "$repo" status --short --untracked-files=all
printf 'TRACKED_DIFF_STAT\n'
git -C "$repo" diff --stat
printf 'TRACKED_DIFF_NUMSTAT\n'
git -C "$repo" diff --numstat
printf 'INDEX_STATE\n'
git -C "$repo" diff --cached --name-status
printf 'GPU_COMPUTE_PROCESSES\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
printf 'RAY_OR_TRAIN_PROCESSES\n'
pgrep -af '[r]aylet|[g]cs_server|[r]ay::|train_embodied_agent' || true
