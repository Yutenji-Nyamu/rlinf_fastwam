#!/usr/bin/env bash
set -u
TZ=Asia/Shanghai date --iso-8601=seconds
id
set +e
gh auth status --hostname github.com
rc=$?
set -e
printf 'gh_auth_status_rc=%s\n' "$rc"
exit 0
