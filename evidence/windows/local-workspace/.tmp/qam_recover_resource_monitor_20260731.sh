set -eu

runtime=/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime
monitor=/root/autodl-tmp/qam_resource_monitor_20260731_v1.sh
driver_pid=$(cat "$runtime/driver.pid")

test -n "$driver_pid"
kill -0 "$driver_pid"
bash -n "$monitor"
test ! -e "$runtime/resources.csv"
test ! -e "$runtime/monitor_recovered.pid"

bash "$monitor" "$driver_pid" "$runtime/resources.csv" 2 \
  >"$runtime/monitor_recovered.log" 2>&1 &
monitor_pid=$!
printf '%s\n' "$monitor_pid" >"$runtime/monitor_recovered.pid"
echo "RECOVERED_MONITOR_PID=$monitor_pid"
echo "RECOVERED_AT=$(TZ=Asia/Shanghai date '+%F %T %Z')"

