set -u

RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
PID=$(cat "$RUN_ROOT/fresh.pid")

date --iso-8601=seconds
if kill -0 "$PID" 2>/dev/null; then
  echo "DRIVER_ALIVE=1"
  ps -o pid=,stat=,etimes=,%cpu=,rss=,args= -p "$PID"
else
  echo "DRIVER_ALIVE=0"
fi

printf 'LOG_BYTES='
stat -c %s "$RUN_ROOT/fresh_driver.log" 2>/dev/null || echo 0
tail -n 30 "$RUN_ROOT/fresh_driver.log" 2>/dev/null || true

echo "== fatal tail =="
grep -Ein 'traceback|out of memory|cuda error|runtimeerror|valueerror|assertionerror|nan|inf|killed|segmentation|fatal|error:' \
  "$RUN_ROOT/fresh_driver.log" 2>/dev/null | tail -n 15 || true

echo "== checkpoint dirs =="
find "$RUN_ROOT" -path '*/checkpoints/global_step_*' -type d -maxdepth 5 -print 2>/dev/null || true

echo "== resources =="
cat "$RUN_ROOT/resource_monitor/fresh/peak.txt" 2>/dev/null || true
tail -n 2 "$RUN_ROOT/resource_monitor/fresh/resources.csv" 2>/dev/null || true

echo "== live gpu =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits

echo "== live cgroup =="
printf 'memory.current='
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
awk '$1 ~ /^(anon|file|shmem|inactive_file|active_file|slab)$/ {print}' /sys/fs/cgroup/memory.stat
