#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
smoke_root=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
runtime=/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/fresh_runtime

printf 'inspect_time\t%s\n' "$(date --iso-8601=seconds)"
printf 'git\t%s\t%s\n' \
  "$(git -C "${repo}" rev-parse HEAD)" \
  "$(git -C "${repo}" rev-list --left-right --count HEAD...@{upstream})"
printf 'status\t%s\n' "$(git -C "${repo}" status --porcelain)"
printf 'processes_begin\n'
pgrep -af \
  'train_embodied_agent|ray::|raylet|gcs_server|robotwin_adjust_bottle_rlt_stage2' \
  | grep -v -E 'pgrep -af|fresh_failure_inspect' \
  || printf '%s\n' NONE
printf 'processes_end\n'
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

printf 'runtime_files_begin\n'
find "${runtime}" -maxdepth 2 -type f -printf '%P\t%s bytes\n' \
  | LC_ALL=C sort
printf 'runtime_files_end\n'
printf 'smoke_files_begin\n'
if test -e "${smoke_root}"; then
  find "${smoke_root}" -maxdepth 4 -type f -printf '%P\t%s bytes\n' \
    | LC_ALL=C sort | head -n 120
else
  printf '%s\n' NONE
fi
printf 'smoke_files_end\n'

for name in started_at.txt finished_at.txt exit_code.txt driver_pid.txt monitor_pid.txt; do
  printf '%s\t' "${name}"
  if test -f "${runtime}/${name}"; then
    tr '\n' ' ' <"${runtime}/${name}"
    printf '\n'
  else
    printf '%s\n' MISSING
  fi
done

printf 'driver_tail_begin\n'
tail -n 160 "${runtime}/driver.log" 2>/dev/null || true
printf 'driver_tail_end\n'
printf 'monitor_tail_begin\n'
tail -n 20 "${runtime}/monitor.log" 2>/dev/null || true
tail -n 8 "${runtime}/resources.csv" 2>/dev/null || true
printf 'monitor_tail_end\n'
printf '%s\n' RLT_FRESH_FAILURE_INSPECT_DONE
