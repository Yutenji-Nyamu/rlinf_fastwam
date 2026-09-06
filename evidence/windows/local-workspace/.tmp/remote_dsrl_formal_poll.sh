set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN_ROOT="$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1"
PID=$(cat "$RUN_ROOT/formal.pid")

echo "TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
if kill -0 "$PID" 2>/dev/null; then
  echo "DRIVER_ALIVE=1 PID=$PID"
else
  echo "DRIVER_ALIVE=0 PID=$PID"
fi
echo "PROCESS_SCAN"
pgrep -af '^/root/autodl-tmp/RLinf/.venv/bin/python .*train_embodied_agent.py|^ray::|raylet|gcs_server' |
  head -n 30 || true
echo "GPU_STATE"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
echo "MEMORY_STATE"
printf "memory.current="
cat /sys/fs/cgroup/memory.current
printf "memory.max="
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
grep -E '^(anon|file|inactive_file|active_file|shmem|slab|kernel_stack|pagetables) ' \
  /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.pressure
echo "RESOURCE_PEAK"
test -s "$RUN_ROOT/resource_monitor/peak.txt" &&
  sed -n '1,80p' "$RUN_ROOT/resource_monitor/peak.txt" || true
echo "RESOURCE_TAIL"
test -s "$RUN_ROOT/resource_monitor/resources.csv" &&
  tail -n 3 "$RUN_ROOT/resource_monitor/resources.csv" || true
echo "CGROUP_DETAIL_TAIL"
test -s "$RUN_ROOT/resource_monitor/cgroup_detail.csv" &&
  tail -n 3 "$RUN_ROOT/resource_monitor/cgroup_detail.csv" || true
echo "LOG_SIGNALS"
grep -E 'global_step|Rollout|rollout|update_step|Training|training|eval|Eval|success|checkpoint|Checkpoint|Traceback|ERROR|Error|Exception|NaN|Inf|CUDA out of memory|OutOfMemory' \
  "$RUN_ROOT/formal_driver.log" | tail -n 80 || true
echo "LOG_TAIL"
tail -n 80 "$RUN_ROOT/formal_driver.log" || true
echo "ARTIFACTS"
find "$RUN_ROOT" -maxdepth 4 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' |
  sort | tail -n 40
du -sh "$RUN_ROOT"
df -h /root/autodl-tmp /dev/shm
