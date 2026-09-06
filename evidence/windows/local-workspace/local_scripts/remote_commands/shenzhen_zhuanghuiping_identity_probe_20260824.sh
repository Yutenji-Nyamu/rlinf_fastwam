#!/usr/bin/env bash
set -euo pipefail
printf 'MARKER=SZ_ZHUANGHUIPING_IDENTITY_PROBE_V1\n'
whoami
id
pwd -P
readlink -f "$HOME/data" "$HOME/shared"
printf 'MARKER=SZ_ZHUANGHUIPING_IDENTITY_PROBE_OK\n'
