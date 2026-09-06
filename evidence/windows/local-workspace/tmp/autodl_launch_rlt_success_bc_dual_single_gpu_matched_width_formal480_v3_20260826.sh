#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
python=/root/autodl-tmp/RLinf/.venv/bin/python
ray_cli=/root/autodl-tmp/RLinf/.venv/bin/ray
driver_script=/tmp/autodl_run_one_rlt_success_bc_formal480_v2_20260825.sh
ray_script=/tmp/autodl_start_shared_ray_formal480_v2_20260825.sh
expected_head=848b61278687702ea717c56b3734f1486cea3b95

control_run=/root/autodl-tmp/experiments/rlt_single_gpu_control_matched_width_formal480_20260826_v3
method_run=/root/autodl-tmp/experiments/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
control_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_control_matched_width_formal480_20260826_v3/runtime
method_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3/runtime
pair_runtime=/root/autodl-tmp/experiment_exports/rlt_success_bc_dual_single_gpu_matched_width_formal480_20260826_v3

for target in "$control_run" "$method_run" "${control_runtime%/runtime}" "${method_runtime%/runtime}" "$pair_runtime"; do
  test ! -e "$target" || { echo "refusing existing target: $target" >&2; exit 20; }
done

actual_head=$(git -C "$repo" rev-parse HEAD)
test "$actual_head" = "$expected_head"
test -z "$(git -C "$repo" status --short)"
test -x "$driver_script"
test -x "$ray_script"
"$python" -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0", 52001)); s.close()'

mkdir -p "$control_runtime" "$method_runtime" "$pair_runtime"
node_ip=$(hostname -I | awk '{print $1}')
ray_address="$node_ip:52001"
setsid "$ray_script" "$node_ip" "$pair_runtime" >"$pair_runtime/ray_head.log" 2>&1 < /dev/null &
ray_head_pid=$!
printf '%s\n' "$ray_head_pid" >"$pair_runtime/ray_head.pid"
printf '%s\n' "$ray_address" >"$pair_runtime/ray_address.txt"

ready=0
for _ in $(seq 1 45); do
  if "$ray_cli" status --address="$ray_address" >"$pair_runtime/ray_status.txt" 2>&1; then ready=1; break; fi
  sleep 2
done
test "$ready" = 1

control_name=robotwin_adjust_bottle_rlt_single_gpu_control_matched_width_formal480_20260826_v3
method_name=robotwin_adjust_bottle_rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3
control_config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_mb256_warm20k_replay80k_control
method_config=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_mb256_warm20k_replay80k_gpu1_fresh480

setsid "$driver_script" 0 "$control_config" "$control_run" "$control_runtime" \
  "$control_name" "$ray_address" rlt_control_mw_f480_v3 \
  >"$control_runtime/foreground.log" 2>&1 < /dev/null &
control_pid=$!
printf '%s\n' "$control_pid" >"$control_runtime/wrapper.pid"

setsid bash -c 'sleep 120; exec "$@"' _ "$driver_script" 1 "$method_config" \
  "$method_run" "$method_runtime" "$method_name" "$ray_address" rlt_method_mw_f480_v3 \
  >"$method_runtime/foreground.log" 2>&1 < /dev/null &
method_pid=$!
printf '%s\n' "$method_pid" >"$method_runtime/wrapper.pid"

monitor_line="printf 'timestamp,memory_current,memory_available_kb,gpu0_mib,gpu0_util,gpu1_mib,gpu1_util\\n' > $(printf '%q' "$pair_runtime/paired_resources.csv"); while kill -0 $control_pid 2>/dev/null || kill -0 $method_pid 2>/dev/null; do ts=\$(date -Is); mc=\$(cat /sys/fs/cgroup/memory.current); ma=\$(awk '/MemAvailable/ {print \$2}' /proc/meminfo); g=\$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\\n' ',' | sed 's/,$//'); printf '%s,%s,%s,%s\\n' \"\$ts\" \"\$mc\" \"\$ma\" \"\$g\" >> $(printf '%q' "$pair_runtime/paired_resources.csv"); sleep 5; done"
setsid bash -lc "$monitor_line" >"$pair_runtime/monitor.log" 2>&1 < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$pair_runtime/monitor.pid"

cleanup_line="while kill -0 $control_pid 2>/dev/null || kill -0 $method_pid 2>/dev/null; do sleep 60; done; kill -INT -- -$ray_head_pid 2>/dev/null || true; sleep 10; kill -TERM -- -$ray_head_pid 2>/dev/null || true; date -Is > $(printf '%q' "$pair_runtime/cleanup_finished_at.txt")"
setsid bash -lc "$cleanup_line" >"$pair_runtime/cleanup.log" 2>&1 < /dev/null &
cleanup_pid=$!
printf '%s\n' "$cleanup_pid" >"$pair_runtime/cleanup.pid"

{
  printf 'source_head=%s\n' "$actual_head"
  printf 'shared_ray_address=%s\n' "$ray_address"
  printf 'shared_ray_head_pid=%s\n' "$ray_head_pid"
  printf 'control_gpu=0\ncontrol_pid=%s\n' "$control_pid"
  printf 'method_gpu=1\nmethod_delayed_pid=%s\n' "$method_pid"
  printf 'method_stagger_seconds=120\n'
  printf 'monitor_pid=%s\ncleanup_pid=%s\n' "$monitor_pid" "$cleanup_pid"
  printf 'control_run=%s\nmethod_run=%s\n' "$control_run" "$method_run"
} >"$pair_runtime/launch_summary.txt"

echo FORMAL_PAIR_V3_LAUNCHED
cat "$pair_runtime/launch_summary.txt"

