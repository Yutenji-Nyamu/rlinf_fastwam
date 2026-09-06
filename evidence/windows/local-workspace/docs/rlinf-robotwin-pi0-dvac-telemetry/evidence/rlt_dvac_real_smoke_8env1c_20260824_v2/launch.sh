#!/usr/bin/env bash
set -euo pipefail

runtime_root=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2/runtime
run_script="${runtime_root}/run_foreground.sh"
monitor=/root/autodl-tmp/tmp/rlt_stage2_resource_monitor_20260729.sh

test -x "${run_script}"
test -f "${monitor}"
test ! -e "${runtime_root}/driver_pid.txt"
if pgrep -af 'train_embodied_agent|ray::|raylet|gcs_server' | grep -v -E 'pgrep -af|rlt_dvac_smoke_launch' >/dev/null; then
  printf 'Refusing smoke while a training/Ray process is active.\n' >&2
  exit 40
fi
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk 'NF')"

nohup "${run_script}" >"${runtime_root}/driver.log" 2>&1 </dev/null &
driver_pid=$!
printf '%s\n' "${driver_pid}" >"${runtime_root}/driver_pid.txt"

nohup bash "${monitor}" "${driver_pid}" "${runtime_root}/resources.csv" 2 \
  >"${runtime_root}/monitor.log" 2>&1 </dev/null &
monitor_pid=$!
printf '%s\n' "${monitor_pid}" >"${runtime_root}/monitor_pid.txt"

sleep 2
kill -0 "${driver_pid}"
printf 'DRIVER_PID=%s\n' "${driver_pid}"
printf 'MONITOR_PID=%s\n' "${monitor_pid}"
printf 'RUNTIME_ROOT=%s\n' "${runtime_root}"
