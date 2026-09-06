#!/usr/bin/env bash
set -euo pipefail

runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime
run=/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1/robotwin_adjust_bottle_rlt_stage2_resource_smoke_8env3c_v1
printf 'resources_header\n'
head -n 1 "$runtime/resources.csv"
printf 'checkpoint_tree_begin\n'
find "$run/checkpoints/global_step_3" -maxdepth 3 -type f \
  -printf '%P\t%s\n' | sort
printf 'checkpoint_tree_end\n'
printf 'small_json_begin\n'
find "$run/checkpoints/global_step_3" -maxdepth 4 -type f \
  \( -name '*.json' -o -name '*.yaml' \) -size -100k -print \
  | sort \
  | while read -r path
do
  printf 'file=%s\n' "$path"
  cat "$path"
  printf '\n'
done
printf 'small_json_end\n'
