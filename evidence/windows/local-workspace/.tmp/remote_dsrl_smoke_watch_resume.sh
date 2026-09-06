set -u

RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
PID=$(cat "$RUN_ROOT/resume.pid")

date --iso-8601=seconds
if kill -0 "$PID" 2>/dev/null; then
  echo "DRIVER_ALIVE=1"
  ps -o pid=,stat=,etimes=,%cpu=,rss=,args= -p "$PID"
else
  echo "DRIVER_ALIVE=0"
fi

printf 'LOG_BYTES='
stat -c %s "$RUN_ROOT/resume_driver.log" 2>/dev/null || echo 0
tail -n 18 "$RUN_ROOT/resume_driver.log" 2>/dev/null || true

echo "== resume markers =="
grep -Ein 'resum|loading local shard|loading dcp|checkpoint|legacy|bitwise|layout mismatch|phase mismatch|shadow.*mismatch|failed to load' \
  "$RUN_ROOT/resume_driver.log" 2>/dev/null | tail -n 25 || true

echo "== fatal tail =="
grep -Ein 'raytaskerror|workercrashederror|actor died|out of memory|cuda error|runtimeerror|valueerror|assertionerror|nan|inf|killed|segmentation|fatal|error:' \
  "$RUN_ROOT/resume_driver.log" 2>/dev/null | tail -n 25 || true

echo "== checkpoint dirs =="
find "$RUN_ROOT" -path '*/checkpoints/global_step_*' -type d -maxdepth 5 -print 2>/dev/null || true

echo "== resources =="
cat "$RUN_ROOT/resource_monitor/resume/peak.txt" 2>/dev/null || true
tail -n 2 "$RUN_ROOT/resource_monitor/resume/resources.csv" 2>/dev/null || true

echo "== live gpu =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits

echo "== live cgroup =="
printf 'memory.current='
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
awk '$1 ~ /^(anon|file|shmem|inactive_file|active_file|slab)$/ {print}' /sys/fs/cgroup/memory.stat
