#!/usr/bin/env bash
set -euo pipefail

key=/home/chenyiteng/.ssh/github_rlinf_fastwam_deploy_ed25519

printf 'MARKER=SZ_GITHUB_DEPLOY_KEY_GENERATE_V1\n'
date --iso-8601=seconds
id

if [[ -e "$key" || -e "$key.pub" ]]; then
  printf 'REFUSE_OVERWRITE_EXISTING_KEY=%s\n' "$key"
  exit 3
fi

umask 077
install -d -m 700 /home/chenyiteng/.ssh
ssh-keygen -q -t ed25519 -N '' -C 'rlinf_fastwam deploy key on SZ-H100 2026-08-22' -f "$key"
chmod 600 "$key"
chmod 644 "$key.pub"

printf 'PRIVATE_KEY_MODE='
stat -c '%a' "$key"
printf 'PUBLIC_KEY_MODE='
stat -c '%a' "$key.pub"
printf 'FINGERPRINT='
ssh-keygen -lf "$key.pub" -E sha256
printf 'PUBLIC_KEY_BEGIN\n'
cat "$key.pub"
printf 'PUBLIC_KEY_END\n'
printf 'MARKER=SZ_GITHUB_DEPLOY_KEY_GENERATE_OK\n'

