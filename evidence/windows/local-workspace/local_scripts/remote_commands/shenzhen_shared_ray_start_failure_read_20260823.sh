#!/usr/bin/env bash
set -euo pipefail

runtime=/data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1
printf '%s\n' '--- files ---'
find "$runtime" -maxdepth 3 -type f -printf '%s %p\n' 2>/dev/null | sort || true
printf '%s\n' '--- ray_start.log ---'
sed -n '1,260p' "$runtime/ray_start.log" 2>/dev/null || true
printf '%s\n' '--- processes ---'
pgrep -u "$(id -u)" -af 'raylet|gcs_server|dashboard' || true
printf '%s\n' '--- ports ---'
ss -ltn | grep -E ':(6389|8266)[[:space:]]' || true
