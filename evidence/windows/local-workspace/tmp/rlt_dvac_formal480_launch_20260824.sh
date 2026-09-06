set -euo pipefail

runtime_root=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
run_script="${runtime_root}/run_foreground.sh"
monitor_script="${runtime_root}/resource_monitor.sh"
test -f "$run_script"
test -f "$monitor_script"
test ! -f "${runtime_root}/started_at.txt"
test ! -f "${runtime_root}/exit_code.txt"
if pgrep -af 'train_embodied_agent.py|ray::|raylet|gcs_server' | grep -v 'pgrep -af' >/dev/null; then
  echo 'existing training or Ray process found' >&2
  exit 1
fi

setsid bash "$run_script" >"${runtime_root}/foreground.log" 2>&1 < /dev/null &
driver_pid=$!
printf '%s\n' "$driver_pid" >"${runtime_root}/driver_pid.txt"
setsid bash "$monitor_script" >"${runtime_root}/monitor.log" 2>&1 < /dev/null &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"${runtime_root}/monitor_pid.txt"
printf 'DRIVER_PID=%s\nMONITOR_PID=%s\n' "$driver_pid" "$monitor_pid"
