#!/usr/bin/env bash
set -euo pipefail

account_key="$HOME/.ssh/github_yutenji_account_ed25519"

date --iso-8601=seconds
hostname
id
printf 'home=%s\n' "$HOME"
pwd
command -v ssh-keygen

if [[ -d "$HOME/.ssh" ]]; then
  stat -c 'ssh_dir mode=%a owner=%U:%G path=%n' "$HOME/.ssh"
  find "$HOME/.ssh" -maxdepth 1 -type f -printf '%f mode=%m owner=%u:%g\n' | sort
else
  printf 'ssh_dir=absent\n'
fi

for path in "$account_key" "$account_key.pub"; do
  if [[ -e "$path" ]]; then
    printf 'target_exists=%s\n' "$path"
    exit 20
  fi
done

printf 'target_private=%s\n' "$account_key"
printf 'target_public=%s.pub\n' "$account_key"
printf 'preflight=PASS\n'
