#!/usr/bin/env bash
set -euo pipefail

run=/root/autodl-tmp/experiments/qam_formal_20260731_v1
exp="$run/robotwin_adjust_bottle_qam_formal_20260731_v1"

printf '=== TOP LEVEL ===\n'
find "$run" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %12s %p\n' | sort

printf '=== CHECKPOINT INVENTORY ===\n'
for step in 25 50; do
  root="$exp/checkpoints/global_step_$step"
  printf '%s\n' "--- global_step_$step ---"
  find "$root" -type f -printf '%12s %P\n' | sort
  printf 'files_bytes='; find "$root" -type f -printf '%s\n' | awk '{n+=1;s+=$1} END {print n+0, s+0}'
  printf 'json_yaml_text\n'
  find "$root" -type f \( -name '*.json' -o -name '*.yaml' -o -name '*.txt' \) -print0 |
    while IFS= read -r -d '' file; do
      printf '%s\n' "--- $file ---"
      sed -n '1,240p' "$file"
    done
done

printf '=== METRICS FILE ===\n'
wc -l -c "$run/metrics.log"
tail -n 12 "$run/metrics.log"
