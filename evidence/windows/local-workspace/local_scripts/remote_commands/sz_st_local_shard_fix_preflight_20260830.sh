#!/usr/bin/env bash
set -euo pipefail

ST_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
PRISM_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo
ST_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
ACTION_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1

printf 'st_head=%s\n' "$(git -C "$ST_WT" rev-parse HEAD)"
printf 'st_branch=%s\n' "$(git -C "$ST_WT" branch --show-current)"
printf 'st_common=%s\n' "$(git -C "$ST_WT" rev-parse --git-common-dir)"
printf 'st_dirty=%s\n' "$(git -C "$ST_WT" status --short | wc -l)"
printf 'prism_patch_is_ancestor=%s\n' "$(git -C "$ST_WT" merge-base --is-ancestor 306ce2e98a06b6f439a1070d8942e20132e48d49 HEAD && echo yes || echo no)"
printf 'st_checkpoint_format_refs=%s\n' "$(grep -c checkpoint_format "$ST_WT/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py" || true)"
printf 'remote_branch='; git -C "$ST_WT" branch -r --contains 306ce2e98a06b6f439a1070d8942e20132e48d49 | sed 's/^[[:space:]]*//' | head -n3 | paste -sd, -; printf '\n'
printf 'worktrees:\n'; git -C "$ST_WT" worktree list --porcelain | sed -n 's/^worktree /  /p'

for run in "$ACTION_RUN" "$ST_RUN"; do
  printf 'run=%s wrapper=' "$run"
  pid=$(cat "$run/runtime/wrapper.pid")
  if kill -0 "$pid" 2>/dev/null; then printf 'alive'; else printf 'dead'; fi
  printf ' pid=%s namespace=' "$pid"
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^RLINF_NAMESPACE=//p' | head -n1 || true
done
printf 'gpu45='; nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu | paste -sd, -; printf '\n'
printf 'gpu67='; nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu | paste -sd, -; printf '\n'
