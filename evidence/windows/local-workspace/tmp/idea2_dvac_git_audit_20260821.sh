#!/usr/bin/env bash
set -u

inspect_repo() {
  repo="$1"
  echo "REPO $repo"
  if ! git -C "$repo" rev-parse --is-inside-work-tree 2>/dev/null; then
    echo NOT_A_GIT_WORKTREE
    return
  fi
  git -C "$repo" status --short --branch
  git -C "$repo" rev-parse HEAD
  git -C "$repo" remote -v
  git -C "$repo" branch -vv --no-abbrev
}

echo RLINF_WORKTREES
git -C /root/autodl-tmp/RLinf worktree list --porcelain

inspect_repo /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
inspect_repo /root/autodl-tmp/RLinf_idea2_dvac_train
inspect_repo /root/autodl-tmp/RLinf

echo OUTER_WORKTREES
git -C /root/autodl-tmp/idea2_dvac_train_wamppo worktree list --porcelain 2>/dev/null || true
inspect_repo /root/autodl-tmp/idea2_dvac_train_wamppo
