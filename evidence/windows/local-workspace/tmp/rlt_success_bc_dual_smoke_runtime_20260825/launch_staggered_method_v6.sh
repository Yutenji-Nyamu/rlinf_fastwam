#!/usr/bin/env bash
set -euo pipefail

package_root=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_runtime_package
control_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_success_bc_pair_smoke_20260825_v9/runtime
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v9
method_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_smoke_20260825_v9/runtime
method_config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_gpu1_fresh480
method_name=robotwin_adjust_bottle_rlt_single_gpu_success_episode_bc_dvac_smoke_v9
method_namespace=RLTSuccessBCSmokeV9
pair_runtime=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_smoke_20260825_pair_v9

test -f "$control_runtime/process_group.txt"
test ! -f "$method_runtime/wrapper.pid"
node_ip=$(hostname -I | awk '{print $1}')
head_pgid=$(cat "$pair_runtime/ray_head.pid")
control_pgid=$(cat "$control_runtime/process_group.txt")

setsid bash "$package_root/run_one_shared_smoke.sh" 1 "$method_config" "$method_run" "$method_runtime" "$method_name" "$node_ip:50001" "$method_namespace" method >"$method_runtime/foreground.log" 2>&1 < /dev/null &
method_pid=$!
printf '%s\n' "$method_pid" >"$method_runtime/wrapper.pid"
sleep 1
method_pgid=$(ps -o pgid= -p "$method_pid" | tr -d ' ')
printf '%s\n' "$method_pgid" >"$method_runtime/process_group.txt"

setsid bash "$package_root/paired_resource_monitor.sh" "$control_runtime" "$method_runtime" "$control_pgid" "$head_pgid" "$method_pgid" "$head_pgid" "$pair_runtime/paired_resources.csv" >"$pair_runtime/monitor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"$pair_runtime/monitor.pid"
setsid bash "$package_root/cleanup_shared_after_both.sh" "$control_runtime" "$method_runtime" "$head_pgid" >"$pair_runtime/cleanup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"$pair_runtime/cleanup.pid"

date -Is >"$pair_runtime/method_launched_at.txt"
cat >>"$pair_runtime/launch_summary.txt" <<EOF
method_gpu=1
method_namespace=$method_namespace
method_pid=$method_pid
method_pgid=$method_pgid
EOF
cat "$pair_runtime/launch_summary.txt"
