set -u

runtime=/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime
driver="$runtime/driver.log"
launcher=/root/autodl-tmp/qam_qonly_smoke_launcher_20260731_v1.log

echo "TIME=$(TZ=Asia/Shanghai date '+%F %T %Z')"
printf 'launcher_alive='
pid=$(cat /root/autodl-tmp/qam_qonly_smoke_launcher_20260731_v1.pid 2>/dev/null || true)
test -n "$pid" && kill -0 "$pid" 2>/dev/null && echo yes || echo no
printf 'driver_alive='
dpid=$(cat "$runtime/driver.pid" 2>/dev/null || true)
test -n "$dpid" && kill -0 "$dpid" 2>/dev/null && echo yes || echo no
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.stat |
  awk '$1=="anon" || $1=="file" {print}'
cat /sys/fs/cgroup/memory.events
stat --format='driver_bytes=%s driver_mtime=%y' "$driver" 2>/dev/null || true
echo "KEY_LOG"
grep -Ei \
  'Built|QAM|replay|global.insert|critic|update|saving checkpoint|success|reward|loss|grad|target|fatal|error|traceback|oom|nan|finished|complete|epoch|step' \
  "$driver" 2>/dev/null |
  tail -100 || true
echo "TAIL"
tail -50 "$driver" 2>/dev/null || true
echo "EXIT"
for file in exit_code.txt monitor_exit_code.txt; do
  test -f "$runtime/$file" && echo "$file=$(cat "$runtime/$file")" || true
done
tail -20 "$launcher" 2>/dev/null || true
echo "RESOURCE_TAIL"
tail -5 "$runtime/resources.csv" 2>/dev/null || true

