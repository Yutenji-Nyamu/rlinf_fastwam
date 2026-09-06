#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_smoke_20260807_v1
run_script="$runtime_root/remote_ogpo_20260807_smoke_run_v1.sh"
monitor_script="$runtime_root/remote_ogpo_20260807_smoke_monitor_v1.sh"
expected_head=5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
expected_resolved_sha=ebd163f647d2a9399fdca007099fac550f6f344392162adc36eede129671f4eb
expected_run_sha=ba3cf334784f4d237febe7edf37bc1104cba997e07351fca1e73e29f4074a9ad
expected_monitor_sha=40e65af8e03a3ebaafa1ff322aee99795832bcd34fafc006f80b8accff0c5368

cd "$repo"
test "$(git rev-parse HEAD)" = "$expected_head"
test "$(git branch --show-current)" = codex/ogpo-pi0-robotwin
test -z "$(git status --short)"
test -f "$runtime_root/resolved.yaml"
test -f "$run_script"
test -f "$monitor_script"
test "$(sha256sum "$runtime_root/resolved.yaml" | awk '{print $1}')" = "$expected_resolved_sha"
test "$(sha256sum "$run_script" | awk '{print $1}')" = "$expected_run_sha"
test "$(sha256sum "$monitor_script" | awk '{print $1}')" = "$expected_monitor_sha"
test ! -e "$runtime_root/started_at.txt"
test ! -e "$runtime_root/driver_pid.txt"
test ! -e "$run_root"
mapfile -t active_rows < <(
  {
    ps -eo pid=,comm=,args= | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
    pgrep -ax raylet || true
    pgrep -ax gcs_server || true
  }
)
test "${#active_rows[@]}" = 0
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF'
)
test "${#compute_rows[@]}" = 0

nohup bash "$run_script" >"$runtime_root/driver.log" 2>&1 </dev/null &
driver_pid=$!
printf '%s\n' "$driver_pid" >"$runtime_root/driver_pid.txt"
nohup bash "$monitor_script" "$driver_pid" "$runtime_root/resources_1s.csv" 1 \
  >"$runtime_root/resource_monitor.log" 2>&1 </dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$runtime_root/monitor_pid.txt"
sleep 3
kill -0 "$driver_pid"
kill -0 "$monitor_pid"

printf 'DRIVER_PID\t%s\n' "$driver_pid"
printf 'MONITOR_PID\t%s\n' "$monitor_pid"
printf 'RUNTIME_ROOT\t%s\n' "$runtime_root"
printf 'RUN_ROOT\t%s\n' "$run_root"
printf '%s\n' OGPO_ROBOTWIN_SMOKE_LAUNCHED
