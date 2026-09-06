#!/usr/bin/env bash
set -euo pipefail

KEY=/home/chenyiteng/.ssh/github_fastwam_deploy_ed25519

test "$(id -un)" = chenyiteng
test ! -e "$KEY"
test ! -e "$KEY.pub"
umask 077
mkdir -p /home/chenyiteng/.ssh
ssh-keygen -q -t ed25519 -N '' -C 'SZ-H100 FastWAM' -f "$KEY"
chmod 600 "$KEY"
chmod 644 "$KEY.pub"

echo '=== FASTWAM_DEPLOY_PUBLIC_KEY ==='
cat "$KEY.pub"
echo '=== FASTWAM_DEPLOY_FINGERPRINT ==='
ssh-keygen -lf "$KEY.pub" -E sha256
echo '=== KEY_MODES ==='
stat -c '%a %U:%G %n' "$KEY" "$KEY.pub"
echo 'FASTWAM_DEPLOY_KEY_PREPARE_OK'
