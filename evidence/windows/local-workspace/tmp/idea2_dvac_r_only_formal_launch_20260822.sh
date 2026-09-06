#!/usr/bin/env bash
set -euo pipefail

run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
launcher="$runtime_dir/launch_formal.sh"

test ! -e "$run_dir"
nohup setsid bash "$launcher" > "$runtime_dir/wrapper.log" 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > "$runtime_dir/wrapper.pid"
printf 'WRAPPER_PID=%s\n' "$wrapper_pid"
printf 'LAUNCH_REQUESTED_AT=%s\n' "$(date --iso-8601=seconds)"
