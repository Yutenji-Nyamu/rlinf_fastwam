#!/usr/bin/env bash
set -euo pipefail

BASE=7d07a4212ee6858cc333e1d4fab7a37256d1f839
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
PPO=examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml

printf 'MARKER=SZ_GRPO_CURRENT_SOURCE_PREFLIGHT_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id

test "$(git -C "$WT" rev-parse HEAD)" = "$BASE"
test "$(git -C "$WT" branch --show-current)" = codex/sz-7d07a421-grpo-pi0-robotwin
test -z "$(git -C "$WT" status --short)"
test -f "$WT/$PPO"
test ! -e "$WT/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml"

printf '%s\n' '=== WORKTREE ==='
git -C "$WT" status --short --branch
git -C "$WT" log -1 --format='head=%H%nsubject=%s'
printf '%s\n' '=== PERSONAL GRPO-LIKE REFS ==='
git -C "$WT" for-each-ref --format='%(refname:short) %(objectname)' refs/remotes/personal/ \
  | awk 'BEGIN{IGNORECASE=1} /grpo|idea2|dvac/ {print}'
printf '%s\n' '=== CURRENT PPO SHA ==='
sha256sum "$WT/$PPO"
printf '%s\n' '=== CURRENT PPO YAML ==='
sed -n '1,260p' "$WT/$PPO"
printf '%s\n' '=== CURRENT GRPO CORE SYMBOLS ==='
grep -R -n -E 'compute_grpo_advantages|compute_grpo_actor_loss_fn|adv_type == "grpo"|filter_rewards|group_size' \
  "$WT/rlinf/algorithms" "$WT/rlinf/workers/actor/embodied_fsdp_actor_worker.py" \
  | sed -n '1,160p'
printf 'MARKER=SZ_GRPO_CURRENT_SOURCE_PREFLIGHT_OK\n'
