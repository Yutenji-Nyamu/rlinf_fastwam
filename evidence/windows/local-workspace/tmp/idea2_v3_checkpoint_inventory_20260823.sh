#!/usr/bin/env bash
set -u
run=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
printf 'CHECKPOINTS\n'
find "$run" -type d -name 'global_step_*' -print0 | sort -zV | while IFS= read -r -d '' path; do
  printf '%s\t' "$path"
  du -sh "$path" | awk '{print $1}'
done
printf 'COMPLETE_MARKERS\n'
find "$run" -type f -name 'complete.json' -print | sort -V
