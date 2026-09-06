#!/usr/bin/env bash
set -euo pipefail

run_dir=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1
driver_log=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/runtime/driver.log

printf 'ALL_RUN_FILES\n'
find "$run_dir" -type f -printf '%s\t%p\n' | sort -n
printf 'EVENT_FILE_COUNT='
find "$run_dir" -type f -name '*tfevents*' | wc -l
printf 'CHECKPOINT_LOG_LINES\n'
grep -aEi \
  'checkpoint|saving|saved|full_weights|global_step_2000' \
  "$driver_log" |
  tail -n 40 || true
