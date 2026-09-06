#!/usr/bin/env bash
set -euo pipefail

repo=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
key=/home/chenyiteng/.ssh/github_rlinf_fastwam_deploy_ed25519

printf 'MARKER=SZ_GITHUB_DEPLOY_KEY_PREFLIGHT_V1\n'
date --iso-8601=seconds
id
printf 'HOME=%s\n' "$HOME"
command -v ssh-keygen

printf '\n[canonical]\n'
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
git -C "$repo" remote -v
git -C "$repo" worktree list --porcelain

printf '\n[target]\n'
printf 'KEY=%s\n' "$key"
if [[ -e "$key" || -e "$key.pub" ]]; then
  printf 'TARGET_EXISTS=1\n'
  ls -l -- "$key" "$key.pub" 2>/dev/null || true
  exit 3
fi
printf 'TARGET_EXISTS=0\n'

if [[ -d /home/chenyiteng/.ssh ]]; then
  printf 'SSH_DIR_EXISTS=1\n'
  find /home/chenyiteng/.ssh -maxdepth 1 -mindepth 1 -printf '%f %m %y\n' | sort
else
  printf 'SSH_DIR_EXISTS=0\n'
fi

