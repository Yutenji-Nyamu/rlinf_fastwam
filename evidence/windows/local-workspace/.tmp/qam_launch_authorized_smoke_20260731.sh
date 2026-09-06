set -eu

launcher=/root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh
launcher_log=/root/autodl-tmp/qam_qonly_smoke_launcher_20260731_v1.log
expected_sha=92a76d47615e34c02f6c33d33fb84d194a03e5310aaf4d3cb0550aeed18d79b4

test "$(sha256sum "$launcher" | awk '{print $1}')" = "$expected_sha"
bash -n "$launcher"
test ! -e "$launcher_log"

nohup bash "$launcher" >"$launcher_log" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" \
  >/root/autodl-tmp/qam_qonly_smoke_launcher_20260731_v1.pid
echo "LAUNCHER_PID=$launcher_pid"
echo "LAUNCH_TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"

