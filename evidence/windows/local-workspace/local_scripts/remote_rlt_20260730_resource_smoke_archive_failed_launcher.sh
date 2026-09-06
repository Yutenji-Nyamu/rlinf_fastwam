#!/usr/bin/env bash
set -euo pipefail

evidence=/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1
failed=/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1_failed_launcher_127
run_root=/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1

test -d "$evidence/runtime"
test "$(cat "$evidence/runtime/exit_code.txt")" = 127
test ! -e "$run_root"
test ! -e "$failed"
test ! -L "$failed"
test -z "$(
  ps -eo pid=,comm=,args= \
    | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
)"
test -z "$(pgrep -ax raylet || true)"
test -z "$(pgrep -ax gcs_server || true)"

mv "$evidence" "$failed"
printf 'archived_failed_launcher\t%s\n' "$failed"
printf 'failure_exit_code\t127\n'
printf 'run_root_created\tno\n'
