#!/usr/bin/env bash
set -euo pipefail

RLINF=/data/chenyiteng/projects/rlinf-shenzhen/RLinf
KEY=/home/chenyiteng/.ssh/github_rlinf_fastwam_deploy_ed25519
KNOWN=/home/chenyiteng/.ssh/github_rlinf_fastwam_known_hosts
URL=git@github.com:Yutenji-Nyamu/rlinf_fastwam.git
SSH_CMD="ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$KNOWN -o StrictHostKeyChecking=yes"

printf 'MARKER=SZ_GITHUB_PERSONAL_REMOTE_ACTIVATE_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id

test "$(git -C "$RLINF" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test -z "$(git -C "$RLINF" status --short)"
test -f "$KEY"
test -f "$KEY.pub"

umask 077
install -d -m 700 /home/chenyiteng/.ssh
printf '%s\n' 'github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl' > "$KNOWN"
chmod 600 "$KNOWN"
printf 'known_hosts_fingerprint='
ssh-keygen -lf "$KNOWN" -E sha256

printf '%s\n' '=== authenticated ls-remote ==='
GIT_SSH_COMMAND="$SSH_CMD" git ls-remote "$URL" HEAD refs/heads/main

if git -C "$RLINF" remote get-url personal >/dev/null 2>&1; then
  test "$(git -C "$RLINF" remote get-url personal)" = "$URL"
else
  git -C "$RLINF" remote add personal "$URL"
fi
git -C "$RLINF" config core.sshCommand "$SSH_CMD"
git -C "$RLINF" fetch --prune personal

printf '%s\n' '=== configured state ==='
git -C "$RLINF" remote -v
git -C "$RLINF" config --get core.sshCommand
GIT_SSH_COMMAND="$SSH_CMD" git ls-remote "$URL" HEAD refs/heads/main
printf 'MARKER=SZ_GITHUB_PERSONAL_REMOTE_ACTIVATE_OK\n'

