#!/usr/bin/env bash
set -euo pipefail

runtime_dir=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
run_dir=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823

test ! -e "$run_dir"
test ! -e "$runtime_dir/wrapper.pid"
chmod 0755 "$runtime_dir/launch_formal.sh" "$runtime_dir/observe_resources.sh"
cd "$runtime_dir"
nohup bash ./launch_formal.sh > ./wrapper.log 2>&1 < /dev/null &
wrapper_pid=$!
printf '%s\n' "$wrapper_pid" > wrapper.pid
printf 'WRAPPER_PID=%s\n' "$wrapper_pid"
printf 'STARTED_AT=%s\n' "$(date --iso-8601=seconds)"
