#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh

BASE=/data/chenyiteng/projects/rlinf-shenzhen
CANONICAL="$BASE/RLinf"
WORKTREES="$BASE/worktrees"
PPO_WORKTREE="$WORKTREES/ppo-pi0-robotwin"
ROBOTWIN="$BASE/RoboTwin-RLinf-support"
RLINF_REV=7d07a4212ee6858cc333e1d4fab7a37256d1f839
ROBOTWIN_REV=0008ae6800df9f75fc8de7098bacb01735fd8fd2

test ! -e "$BASE"
mkdir -p "$BASE"

timeout --signal=INT --kill-after=60s 1800s \
  git clone https://github.com/RLinf/RLinf.git "$CANONICAL"
test "$(git -C "$CANONICAL" rev-parse refs/remotes/origin/main)" = "$RLINF_REV"
git -C "$CANONICAL" checkout --detach "$RLINF_REV"

mkdir -p "$WORKTREES"
git -C "$CANONICAL" worktree add -b codex/sz-ppo-pi0-robotwin "$PPO_WORKTREE" "$RLINF_REV"

timeout --signal=INT --kill-after=60s 1800s \
  git clone --branch RLinf_support --single-branch \
    https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN"
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = "$ROBOTWIN_REV"
git -C "$ROBOTWIN" checkout --detach "$ROBOTWIN_REV"

printf '%s\n' '=== SOURCE MANIFEST ==='
printf 'canonical_head='
git -C "$CANONICAL" rev-parse HEAD
printf 'canonical_status='
git -C "$CANONICAL" status --short --branch
printf 'ppo_head='
git -C "$PPO_WORKTREE" rev-parse HEAD
printf 'ppo_branch='
git -C "$PPO_WORKTREE" branch --show-current
printf 'ppo_status='
git -C "$PPO_WORKTREE" status --short --branch
printf 'robotwin_head='
git -C "$ROBOTWIN" rev-parse HEAD
printf 'robotwin_status='
git -C "$ROBOTWIN" status --short --branch
git -C "$CANONICAL" worktree list --porcelain
du -sh "$CANONICAL" "$PPO_WORKTREE" "$ROBOTWIN"
df -h /data
