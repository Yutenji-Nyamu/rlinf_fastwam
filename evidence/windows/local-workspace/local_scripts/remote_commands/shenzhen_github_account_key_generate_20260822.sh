#!/usr/bin/env bash
set -euo pipefail

account_key="$HOME/.ssh/github_yutenji_account_ed25519"

date --iso-8601=seconds
hostname
id

install -d -m 700 "$HOME/.ssh"
for path in "$account_key" "$account_key.pub"; do
  if [[ -e "$path" ]]; then
    printf 'refusing_existing_target=%s\n' "$path"
    exit 20
  fi
done

umask 077
ssh-keygen -q -t ed25519 -N '' -C 'SZ-H100 GitHub account 2026-08-22' -f "$account_key"
chmod 600 "$account_key"
chmod 644 "$account_key.pub"

stat -c 'private mode=%a owner=%U:%G path=%n' "$account_key"
stat -c 'public mode=%a owner=%U:%G path=%n' "$account_key.pub"
ssh-keygen -lf "$account_key.pub" -E sha256
printf 'public_key=' 
cat "$account_key.pub"
printf 'generation=PASS\n'
