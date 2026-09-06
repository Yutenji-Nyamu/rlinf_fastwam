#!/usr/bin/env bash
set -euo pipefail

printf 'MARKER=SZ_GITHUB_GH_WEB_LOGIN_DEVICE_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id
export GH_BROWSER=echo
printf '\n' | timeout --foreground 600s gh auth login --hostname github.com --web
printf 'MARKER=SZ_GITHUB_GH_WEB_LOGIN_DEVICE_OK\n'
