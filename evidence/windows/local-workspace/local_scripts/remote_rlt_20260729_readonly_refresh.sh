#!/usr/bin/env bash
set -u

base_repo=/root/autodl-tmp/RLinf
dsrl_worktree=/root/autodl-tmp/RLinf_fastwam_rlinf
rlt_worktree=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
rlt_base=48a775db09c16c455aeba7b0600c920e7c80d534

echo '[identity]'
date -Is
hostname
id -u

echo '[resources]'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
free -h
df -hT /root/autodl-tmp

echo '[training_processes]'
pgrep -af 'ray::|raylet|gcs_server|train_embodied_agent|RoboTwin|robotwin|torchrun' || true

echo '[base_repo]'
git -C "$base_repo" branch --show-current
git -C "$base_repo" rev-parse HEAD
git -C "$base_repo" status --short
git -C "$base_repo" worktree list --porcelain
git -C "$base_repo" remote

echo '[requested_base]'
git -C "$base_repo" cat-file -t "$rlt_base"
git -C "$base_repo" show -s --format='%H%n%P%n%ad%n%s' --date=iso-strict "$rlt_base"
git -C "$base_repo" branch --all --contains "$rlt_base"
git -C "$base_repo" merge-base "$rlt_base" HEAD
git -C "$base_repo" merge-base --is-ancestor "$rlt_base" HEAD
echo "ancestor_of_base_head_exit=$?"

echo '[dsrl_worktree]'
if test -d "$dsrl_worktree"; then
  git -C "$dsrl_worktree" branch --show-current
  git -C "$dsrl_worktree" rev-parse HEAD
  git -C "$dsrl_worktree" status --short
else
  echo 'missing'
fi

echo '[rlt_target_collision_check]'
if test -e "$rlt_worktree"; then
  echo "path_exists=$rlt_worktree"
else
  echo 'path_absent'
fi
if git -C "$base_repo" show-ref --verify --quiet refs/heads/codex/rlt-pi0-robotwin; then
  echo 'local_branch_exists'
else
  echo 'local_branch_absent'
fi
if git -C "$base_repo" show-ref --verify --quiet refs/remotes/personal/codex/rlt-pi0-robotwin; then
  echo 'personal_remote_branch_exists'
else
  echo 'personal_remote_branch_absent'
fi

echo '[key_paths]'
for path in \
  /root/autodl-tmp/RLinf \
  /root/autodl-tmp/RLinf_fastwam_rlinf \
  /root/autodl-tmp/RoboTwin_RLinf \
  /root/autodl-tmp/models \
  /root/autodl-tmp/experiment_exports \
  /root/autodl-tmp/datasets \
  /root/autodl-tmp/cache \
  /root/autodl-tmp/.cache
do
  if test -e "$path"; then
    stat -c '%F %n' "$path"
  else
    echo "missing $path"
  fi
done
