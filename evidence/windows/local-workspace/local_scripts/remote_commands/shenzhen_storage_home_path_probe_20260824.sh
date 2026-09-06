#!/usr/bin/env bash
set -euo pipefail

printf 'MARKER=SZ_HOME_STORAGE_PATH_PROBE_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
printf '%s\n' '=== FILESYSTEMS ==='
df -hT / /home /data /home/zhuanghuiping
printf '%s\n' '=== MOUNTS ==='
findmnt -T / -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,USE%
findmnt -T /home -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,USE%
findmnt -T /data -o TARGET,SOURCE,FSTYPE,SIZE,USED,AVAIL,USE%
printf '%s\n' '=== USER PATHS ==='
for path in /home/zhuanghuiping/datasets /home/zhuanghuiping/data /data/zhuanghuiping; do
  if [[ -e "$path" || -L "$path" ]]; then
    printf 'path=%s resolved=%s\n' "$path" "$(readlink -f "$path")"
    df -hT "$path" | tail -n 1
    du -sh "$path" 2>/dev/null || true
  else
    printf 'path=%s absent\n' "$path"
  fi
done
printf 'MARKER=SZ_HOME_STORAGE_PATH_PROBE_OK\n'
