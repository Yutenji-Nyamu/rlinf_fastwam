set -u

RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
PID_FILE="$RUN_ROOT/fresh.pid"
DRIVER_LOG="$RUN_ROOT/fresh_driver.log"

echo "== poll =="
date --iso-8601=seconds
if test -f "$PID_FILE"; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "DRIVER_ALIVE=1 PID=$PID"
    ps -o pid=,ppid=,stat=,etimes=,%cpu=,%mem=,rss=,args= -p "$PID"
  else
    echo "DRIVER_ALIVE=0 PID=$PID"
  fi
else
  echo "PID_FILE_MISSING=1"
fi

echo "== target processes =="
ps -eo pid=,ppid=,comm=,stat=,etimes=,%cpu=,%mem=,rss=,args= --sort=pid |
  awk '$3 ~ /^(python|python3|raylet|gcs_server)$/ && $0 ~ /(train_embodied_agent|ray::|raylet|gcs_server|monitor_resources.py)/ {print}'

echo "== log tail =="
if test -f "$DRIVER_LOG"; then
  wc -c "$DRIVER_LOG"
  tail -n 100 "$DRIVER_LOG"
fi

echo "== fatal scan =="
if test -f "$DRIVER_LOG"; then
  grep -Ein 'traceback|out of memory|cuda error|runtimeerror|valueerror|assertionerror|nan|inf|killed|segmentation|fatal|error:' "$DRIVER_LOG" | tail -n 40 || true
fi

echo "== output files =="
find "$RUN_ROOT" -maxdepth 4 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null | sort | tail -n 80
if test -f "$RUN_ROOT/resource_monitor/fresh/peak.txt"; then
  echo "== resource peaks =="
  cat "$RUN_ROOT/resource_monitor/fresh/peak.txt"
fi
if test -f "$RUN_ROOT/resource_monitor/fresh/resources.csv"; then
  echo "== latest resource samples =="
  tail -n 4 "$RUN_ROOT/resource_monitor/fresh/resources.csv"
fi

echo "== gpu =="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits

echo "== cgroup =="
printf 'memory.current='
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
awk '$1 ~ /^(anon|file|shmem|file_mapped|inactive_file|active_file|kernel_stack|pagetables|slab)$/ {print}' /sys/fs/cgroup/memory.stat
test ! -f /sys/fs/cgroup/memory.pressure || cat /sys/fs/cgroup/memory.pressure
