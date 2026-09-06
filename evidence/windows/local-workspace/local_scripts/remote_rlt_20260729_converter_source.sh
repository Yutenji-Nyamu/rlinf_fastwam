#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/RoboTwin/policy/pi0

for path in \
  "$root/scripts/process_data.py" \
  "$root/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py" \
  "$root/examples/aloha_real/convert_aloha_data_to_lerobot.py"
do
  printf '=== %s ===\n' "$path"
  if [[ -f "$path" ]]; then
    sha256sum "$path"
    sed -n '1,280p' "$path"
  else
    printf '%s\n' 'MISSING'
  fi
done
