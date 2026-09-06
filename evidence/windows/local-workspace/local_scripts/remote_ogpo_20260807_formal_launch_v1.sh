#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1
run_script="$runtime_root/remote_ogpo_20260807_formal_run_v1.sh"
monitor_script="$runtime_root/remote_ogpo_20260807_formal_monitor_v1.sh"
expected_head=5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
expected_resolved_sha=77419258766880fca7b8d15d725dfb9f2fba5b88d2bfe21eb918fd760e9a7bc9
expected_exact_command_sha=1c5d3aed7a8d73a82cd67ee066b799d77c9180b33196f6683bdfa18eecd95888
expected_provenance_sha=c26895476dc30023916f9a2bfad1b768a88d4f27af0385a0afb036e7a2d84a9d
expected_stop_conditions_sha=642a9b7ccf57b23d6bc6e13e800d760bb3e25bdb552cbdd8e2cdb8d6480e3434
expected_run_sha=c61a712d1aca5d3a717b462f869d4c8689e42406fbe505d6f5a59c3afbffe0a2
expected_monitor_sha=e7c16a18561224c3915686291d48ffc9cbaf795d0fa6e259f7913a2303e9dad5

cd "$repo"
test "$(git rev-parse HEAD)" = "$expected_head"
test "$(git branch --show-current)" = codex/ogpo-pi0-robotwin
test -z "$(git status --short)"
test "$(git rev-list --left-right --count HEAD...@{upstream})" = $'0\t0'
test "$(sha256sum "$runtime_root/resolved.yaml" | awk '{print $1}')" = "$expected_resolved_sha"
test "$(sha256sum "$runtime_root/exact_command.txt" | awk '{print $1}')" = "$expected_exact_command_sha"
test "$(sha256sum "$runtime_root/run_provenance.tsv" | awk '{print $1}')" = "$expected_provenance_sha"
test "$(sha256sum "$runtime_root/stop_conditions.txt" | awk '{print $1}')" = "$expected_stop_conditions_sha"
test "$(sha256sum "$run_script" | awk '{print $1}')" = "$expected_run_sha"
test "$(sha256sum "$monitor_script" | awk '{print $1}')" = "$expected_monitor_sha"
test ! -e "$runtime_root/launched_at.txt"
test ! -e "$runtime_root/started_at.txt"
test ! -e "$runtime_root/driver_pid.txt"
test ! -e "$runtime_root/monitor_pid.txt"
test ! -e "$run_root"

mapfile -t active_rows < <(
  {
    ps -eo pid=,comm=,args= \
      | awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/ {print}'
    pgrep -ax raylet || true
    pgrep -ax gcs_server || true
  }
)
test "${#active_rows[@]}" = 0
mapfile -t compute_rows < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF'
)
test "${#compute_rows[@]}" = 0

date --iso-8601=seconds >"$runtime_root/launched_at.txt"
nohup bash "$run_script" >"$runtime_root/driver.log" 2>&1 </dev/null &
driver_pid=$!
printf '%s\n' "$driver_pid" >"$runtime_root/driver_pid.txt"
nohup bash "$monitor_script" "$driver_pid" "$runtime_root/resources_1s.csv" 1 \
  "$runtime_root/monitor_exit_code.txt" \
  >"$runtime_root/resource_monitor.log" 2>&1 </dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$runtime_root/monitor_pid.txt"
sleep 5
kill -0 "$driver_pid"
kill -0 "$monitor_pid"

printf 'LAUNCHED_AT\t%s\n' "$(cat "$runtime_root/launched_at.txt")"
printf 'DRIVER_PID\t%s\n' "$driver_pid"
printf 'MONITOR_PID\t%s\n' "$monitor_pid"
printf 'RUNTIME_ROOT\t%s\n' "$runtime_root"
printf 'RUN_ROOT\t%s\n' "$run_root"
printf '%s\n' OGPO_ROBOTWIN_FORMAL_LAUNCHED
