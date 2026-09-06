#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
stage=/root/autodl-tmp/qam_formal_docs_sync_20260731_v1
file=docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md

test "$(git -C "$repo" rev-parse HEAD)" = \
  d6fa0f0f4915587ae5e6a03c580fea7938acd3ca
test -z "$(git -C "$repo" status --porcelain=v1)"
printf '%s  %s\n' \
  b80e31dc0ae12e6d207a3b0d336314e61ebe2766914f4f5d4eb8663bb8c0bec6 \
  "$stage/$file" | sha256sum -c -
install -D -m 0644 "$stage/$file" "$repo/$file"
git -C "$repo" diff --check -- "$file"
if grep -n -F 'NBo7SQqoatnZ' "$repo/$file"; then
  printf 'SECRET_SCAN_FAIL=password\n' >&2
  exit 1
fi
git -C "$repo" add -- "$file"
test "$(git -C "$repo" diff --cached --name-only)" = "$file"
git -C "$repo" diff --cached --check
git -C "$repo" commit -m 'docs(qam): record formal evidence push'
printf 'LEDGER_COMMIT='
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
