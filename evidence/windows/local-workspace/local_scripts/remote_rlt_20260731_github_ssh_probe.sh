#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

printf 'SSH_FILES_BEGIN\n'
if [[ -d /root/.ssh ]]; then
  find /root/.ssh -maxdepth 1 -type f \
    -printf '%f mode=%m bytes=%s\n' | sort
else
  printf '<no-root-ssh-dir>\n'
fi
printf 'SSH_FILES_END\n'

set +e
ssh_output="$(
  timeout 20 ssh -T -p 443 \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    -o ConnectionAttempts=1 \
    -o StrictHostKeyChecking=yes \
    git@ssh.github.com 2>&1
)"
ssh_rc=$?
set -e
printf 'SSH_RC=%s\n' "$ssh_rc"
printf '%s\n' "$ssh_output" | sed -E \
  's/(token|password|secret|authorization)[^[:space:]]*/<redacted>/Ig'
