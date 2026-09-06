#!/usr/bin/env bash
set -euo pipefail
WORKTREE=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
git -C "$WORKTREE" diff --cached --stat
printf '%s\n' '===== symbols ====='
git -C "$WORKTREE" grep -n -E \
  'rlt_train_vla|rlt_action_adapter|decode_rlt_action|FullTaskRLTRoute|rlt_transition_replay|model_actions|rtc_enabled' \
  -- rlinf/models/embodiment/openpi rlinf/algorithms/rlt
printf '%s\n' '===== expert ref correction ====='
git -C "$WORKTREE" grep -n -A12 -B4 'ref_actions = ref_chunk.reshape' -- rlinf/algorithms/rlt/route.py
printf '%s\n' '===== no unexpected files ====='
git -C "$WORKTREE" status --short
