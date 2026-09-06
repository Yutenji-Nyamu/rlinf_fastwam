#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
python=/root/autodl-tmp/RLinf/.venv/bin/python
ray_cli=/root/autodl-tmp/RLinf/.venv/bin/ray
driver=/tmp/autodl_run_one_rlt_pure_formal480_20260828.sh
ray_start=/tmp/autodl_start_shared_ray_pure_formal480_20260828.sh
expected_head=a2ae5cbe81049fb7027c43ea85483ed3ffc3ce2f

s05_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1
s20_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1
s05_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1/runtime
s20_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1/runtime
pair_runtime=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_single_gpu_formal480_20260828_v1

for p in "$s05_run" "$s20_run" "${s05_runtime%/runtime}" "${s20_runtime%/runtime}" "$pair_runtime"; do test ! -e "$p" || exit 20; done
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --short)"
chmod +x "$driver" "$ray_start"
mkdir -p "$s05_runtime" "$s20_runtime" "$pair_runtime"

node_ip=$(hostname -I | awk '{print $1}')
ray_address="$node_ip:52001"
setsid "$ray_start" "$node_ip" "$pair_runtime" >"$pair_runtime/ray_head.log" 2>&1 < /dev/null &
ray_pid=$!
printf '%s\n' "$ray_pid" >"$pair_runtime/ray_head.pid"
for _ in $(seq 1 45); do "$ray_cli" status --address="$ray_address" >"$pair_runtime/ray_status.txt" 2>&1 && break; sleep 2; done
"$ray_cli" status --address="$ray_address" >/dev/null

s05_cfg=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s0p5_mb256_warm20k_replay80k_fresh480
s20_cfg=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s2p0_mb256_warm20k_replay80k_gpu1_fresh480
s05_name=robotwin_adjust_bottle_rlt_single_gpu_dvac_pure_reference_bc_s0p5_formal480_20260828_v1
s20_name=robotwin_adjust_bottle_rlt_single_gpu_dvac_pure_reference_bc_s2p0_formal480_20260828_v1

setsid "$driver" "$s05_cfg" "$s05_run" "$s05_runtime" "$s05_name" "$ray_address" rlt_pure_s05_f480_v1 >"$s05_runtime/foreground.log" 2>&1 < /dev/null &
s05_pid=$!
printf '%s\n' "$s05_pid" >"$s05_runtime/wrapper.pid"
setsid bash -c 'sleep 120; exec "$@"' _ "$driver" "$s20_cfg" "$s20_run" "$s20_runtime" "$s20_name" "$ray_address" rlt_pure_s20_f480_v1 >"$s20_runtime/foreground.log" 2>&1 < /dev/null &
s20_pid=$!
printf '%s\n' "$s20_pid" >"$s20_runtime/wrapper.pid"

setsid bash -lc "printf 'timestamp,memory_current,memory_available_kb,gpu0_mib,gpu0_util,gpu1_mib,gpu1_util\\n' > '$pair_runtime/paired_resources.csv'; while kill -0 $s05_pid 2>/dev/null || kill -0 $s20_pid 2>/dev/null; do ts=\$(date -Is); mc=\$(cat /sys/fs/cgroup/memory.current); ma=\$(awk '/MemAvailable/ {print \$2}' /proc/meminfo); g=\$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\\n' ',' | sed 's/,$//'); printf '%s,%s,%s,%s\\n' \"\$ts\" \"\$mc\" \"\$ma\" \"\$g\" >> '$pair_runtime/paired_resources.csv'; sleep 5; done" >"$pair_runtime/monitor.log" 2>&1 < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$pair_runtime/monitor.pid"
setsid bash -lc "while kill -0 $s05_pid 2>/dev/null || kill -0 $s20_pid 2>/dev/null; do sleep 60; done; kill -INT -- -$ray_pid 2>/dev/null || true; sleep 10; kill -TERM -- -$ray_pid 2>/dev/null || true; date -Is > '$pair_runtime/cleanup_finished_at.txt'" >"$pair_runtime/cleanup.log" 2>&1 < /dev/null &
cleanup_pid=$!
printf '%s\n' "$cleanup_pid" >"$pair_runtime/cleanup.pid"

printf 'source_head=%s\nray_address=%s\nray_pid=%s\ns05_gpu=0\ns05_pid=%s\ns20_gpu=1\ns20_pid=%s\nstagger_seconds=120\nmonitor_pid=%s\ncleanup_pid=%s\n' "$expected_head" "$ray_address" "$ray_pid" "$s05_pid" "$s20_pid" "$monitor_pid" "$cleanup_pid" >"$pair_runtime/launch_summary.txt"
cat "$pair_runtime/launch_summary.txt"
