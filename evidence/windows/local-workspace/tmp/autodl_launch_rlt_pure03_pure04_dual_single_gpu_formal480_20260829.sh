#!/usr/bin/env bash
set -euo pipefail

expected_head="$1"
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
ray_cli=/root/autodl-tmp/RLinf/.venv/bin/ray
driver=/tmp/autodl_run_one_rlt_pure_formal480_20260828.sh
ray_start=/tmp/autodl_start_shared_ray_pure_formal480_20260828.sh

p03_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1
p04_run=/root/autodl-tmp/experiments/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1
p03_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1/runtime
p04_runtime=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1/runtime
pair_runtime=/root/autodl-tmp/experiment_exports/rlt_dvac_pure03_pure04_dual_single_gpu_formal480_20260829_v1

test "$(git -C "$repo" rev-parse HEAD)" = "$expected_head"
test -z "$(git -C "$repo" status --short)"
for path in "$p03_run" "$p04_run" "${p03_runtime%/runtime}" "${p04_runtime%/runtime}" "$pair_runtime"; do
  test ! -e "$path" || { printf 'target_exists=%s\n' "$path"; exit 20; }
done
if ps -eo args | grep -E 'train_embodied_agent.py|ray_shared_52001|gcs_server.*52001|raylet.*52001' | grep -v grep >/dev/null; then
  printf '%s\n' 'old_training_or_ray_process_still_present'
  ps -eo pid,pgid,stat,args | grep -E 'train_embodied_agent.py|ray_shared_52001|gcs_server.*52001|raylet.*52001' | grep -v grep || true
  exit 21
fi
chmod +x "$driver" "$ray_start"
mkdir -p "$p03_runtime" "$p04_runtime" "$pair_runtime"

node_ip=$(hostname -I | awk '{print $1}')
ray_address="$node_ip:52001"
setsid "$ray_start" "$node_ip" "$pair_runtime" >"$pair_runtime/ray_head.log" 2>&1 < /dev/null &
ray_pid=$!
printf '%s\n' "$ray_pid" >"$pair_runtime/ray_head.pid"
for _ in $(seq 1 45); do "$ray_cli" status --address="$ray_address" >"$pair_runtime/ray_status.txt" 2>&1 && break; sleep 2; done
"$ray_cli" status --address="$ray_address" >/dev/null

p03_cfg=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s1p0_mb256_warm20k_replay80k_fresh480
p04_cfg=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s1p5_mb256_warm20k_replay80k_gpu1_fresh480
p03_name=robotwin_adjust_bottle_rlt_single_gpu_dvac_pure03_reference_bc_s1p0_formal480_20260829_v1
p04_name=robotwin_adjust_bottle_rlt_single_gpu_dvac_pure04_reference_bc_s1p5_formal480_20260829_v1

setsid "$driver" "$p03_cfg" "$p03_run" "$p03_runtime" "$p03_name" "$ray_address" rlt_pure03_s10_f480_v1 >"$p03_runtime/foreground.log" 2>&1 < /dev/null &
p03_pid=$!
printf '%s\n' "$p03_pid" >"$p03_runtime/wrapper.pid"
setsid bash -c 'sleep 120; exec "$@"' _ "$driver" "$p04_cfg" "$p04_run" "$p04_runtime" "$p04_name" "$ray_address" rlt_pure04_s15_f480_v1 >"$p04_runtime/foreground.log" 2>&1 < /dev/null &
p04_pid=$!
printf '%s\n' "$p04_pid" >"$p04_runtime/wrapper.pid"

setsid bash -lc "printf 'timestamp,memory_current,memory_available_kb,gpu0_mib,gpu0_util,gpu1_mib,gpu1_util\\n' > '$pair_runtime/paired_resources.csv'; while kill -0 $p03_pid 2>/dev/null || kill -0 $p04_pid 2>/dev/null; do ts=\$(date -Is); mc=\$(cat /sys/fs/cgroup/memory.current); ma=\$(awk '/MemAvailable/ {print \$2}' /proc/meminfo); g=\$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\\n' ',' | sed 's/,$//'); printf '%s,%s,%s,%s\\n' \"\$ts\" \"\$mc\" \"\$ma\" \"\$g\" >> '$pair_runtime/paired_resources.csv'; sleep 5; done" >"$pair_runtime/monitor.log" 2>&1 < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$pair_runtime/monitor.pid"
setsid bash -lc "while kill -0 $p03_pid 2>/dev/null || kill -0 $p04_pid 2>/dev/null; do sleep 60; done; kill -INT -- -$ray_pid 2>/dev/null || true; sleep 10; kill -TERM -- -$ray_pid 2>/dev/null || true; date -Is > '$pair_runtime/cleanup_finished_at.txt'" >"$pair_runtime/cleanup.log" 2>&1 < /dev/null &
cleanup_pid=$!
printf '%s\n' "$cleanup_pid" >"$pair_runtime/cleanup.pid"

printf 'source_head=%s\nray_address=%s\nray_pid=%s\npure03_gpu=0\npure03_pid=%s\npure04_gpu=1\npure04_pid=%s\nstagger_seconds=120\nmonitor_pid=%s\ncleanup_pid=%s\n' "$expected_head" "$ray_address" "$ray_pid" "$p03_pid" "$p04_pid" "$monitor_pid" "$cleanup_pid" >"$pair_runtime/launch_summary.txt"
cat "$pair_runtime/launch_summary.txt"
