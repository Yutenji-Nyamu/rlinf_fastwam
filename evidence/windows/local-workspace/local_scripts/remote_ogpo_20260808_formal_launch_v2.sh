#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
runtime_root=/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime
run_root=/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260808_v2
run_script="$runtime_root/remote_ogpo_20260808_formal_run_v2.sh"
monitor_script="$runtime_root/remote_ogpo_20260808_formal_monitor_v2.sh"
expected_head=5d5c84e3ac4efa1713a4139a05ac1b776e634ed3
expected_resolved_sha=352f8e80752d60624a0c53c62d21dcc10bdc8e6712a433c18d0eec0ee1f56a36
expected_exact_command_sha=0faa8cdd976a400aec6ff48cff3c1c0e3b0336e3e885b77eacef00b0631452f6
expected_provenance_sha=4470be3551d00fbc4285c897be6c11390a9cfc03402cff744aca8fd386abf8a3
expected_stop_conditions_sha=2eaf6e5dee226acdc891f0149553a012cb2e5f54d18111bfa44f2bba0171f1d2
expected_run_sha=ed694eeadae6e6e8506bee45aeb500a20f2a9402f2db7eab60c4ba12aa556ee3
expected_monitor_sha=505665dab6f3afc2082b156e7c6296a4f4002a1132e566624a886e500acdc0a1

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
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF')"

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
printf '%s\n' OGPO_ROBOTWIN_FORMAL_V2_LAUNCHED
