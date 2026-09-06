#!/usr/bin/env bash
set -euo pipefail

printf 'MARKER=SZ_HOME_README_INVENTORY_ADMIN_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id
sudo -S -k -p '' /bin/bash -c '
set -euo pipefail
printf "%s\n" "=== HOME ROOT ==="
stat -c "%A %U:%G %s %n" /home
printf "%s\n" "=== README FILES ==="
find /home -xdev -maxdepth 4 -type f \( -iname "README" -o -iname "README.*" -o -iname "*README*.md" -o -iname "*README*.txt" \) -printf "%m %u:%g %s %p\n" | sort
printf "%s\n" "=== ROOT-LEVEL README CONTENT ==="
find /home -xdev -maxdepth 1 -type f \( -iname "README" -o -iname "README.*" -o -iname "*README*.md" -o -iname "*README*.txt" \) -print0 | while IFS= read -r -d "" f; do
  printf "%s\n" "--- $f ---"
  sed -n "1,220p" "$f"
done
'
printf 'MARKER=SZ_HOME_README_INVENTORY_ADMIN_OK\n'
