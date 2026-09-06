#!/usr/bin/env bash
set -u

src=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
target=/root/autodl-tmp/RLinf_rlt_teacher_dvac

printf 'NOW=%s\n' "$(date --iso-8601=seconds)"
printf 'IDENTITY\n'
hostname
pwd
id -u

printf 'RLT_BASE\n'
if [ -d "$src" ]; then
  printf 'PATH=%s\n' "$src"
  git -C "$src" rev-parse HEAD
  git -C "$src" branch --show-current
  git -C "$src" status --short --branch
  printf 'UPSTREAM='
  git -C "$src" rev-parse --abbrev-ref '@{u}' 2>/dev/null || true
  printf 'AHEAD_BEHIND='
  git -C "$src" rev-list --left-right --count '@{u}...HEAD' 2>/dev/null || true
  printf 'REMOTES\n'
  git -C "$src" remote -v
else
  printf 'MISSING=%s\n' "$src"
fi

printf 'TARGET_EXISTS=%s\n' "$([ -e "$target" ] && echo 1 || echo 0)"
printf 'WORKTREES\n'
git -C "$src" worktree list --porcelain 2>/dev/null || true

printf 'WORKTREE_STATUS_AND_SYNC\n'
git -C "$src" worktree list --porcelain 2>/dev/null | awk '/^worktree / {print substr($0,10)}' | while IFS= read -r wt; do
  [ -d "$wt" ] || continue
  printf '\n--- %s ---\n' "$wt"
  printf 'HEAD='; git -C "$wt" rev-parse --short=12 HEAD 2>/dev/null || true
  printf 'BRANCH='; git -C "$wt" branch --show-current 2>/dev/null || true
  git -C "$wt" status --short --branch 2>/dev/null | head -n 80 || true
  up=$(git -C "$wt" rev-parse --abbrev-ref '@{u}' 2>/dev/null || true)
  printf 'UPSTREAM=%s\n' "$up"
  if [ -n "$up" ]; then
    printf 'UPSTREAM_AHEAD_BEHIND='; git -C "$wt" rev-list --left-right --count "$up...HEAD" 2>/dev/null || true
  fi
done

printf 'REMOTE_RLT_BRANCH\n'
timeout 20 git -C "$src" ls-remote personal refs/heads/codex/rlt-pi0-robotwin 2>&1 || true

printf 'RLT_SOURCE_SYMBOLS\n'
rg -n "class RLT|RLT_OBS_KEYS|rlt_|actor_loss|qf_pi|CrossQ|state_dict|load_state_dict|warmup_min_size|compact" \
  "$src/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py" \
  "$src/rlinf/algorithms/rlt" \
  "$src/rlinf/models/embodiment/openpi/openpi_action_model.py" \
  "$src/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml" 2>/dev/null | head -n 260 || true

printf 'RESOURCE_SNAPSHOT\n'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
printf 'memory.current='; cat /sys/fs/cgroup/memory.current
printf 'memory.max='; cat /sys/fs/cgroup/memory.max
grep -E '^max |^oom |^oom_kill ' /sys/fs/cgroup/memory.events
free -h
df -h /root/autodl-tmp | tail -n 1

