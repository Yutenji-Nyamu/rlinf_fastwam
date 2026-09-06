set -u

echo '=== IDENTITY ==='
hostname
pwd
id -u
date '+%F %T %Z'

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
cd "$REPO"

echo '=== PROCESSES ==='
pgrep -af 'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker|raylet|gcs_server' || true

echo '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits

echo '=== HOST_MEMORY ==='
free -h
echo '=== CGROUP_MEMORY ==='
if test -r /sys/fs/cgroup/memory.current; then
  printf 'memory.current='; cat /sys/fs/cgroup/memory.current
fi
if test -r /sys/fs/cgroup/memory.max; then
  printf 'memory.max='; cat /sys/fs/cgroup/memory.max
fi
cat /sys/fs/cgroup/memory.events 2>/dev/null || true

echo '=== RECENT_RUNS ==='
find "$REPO/logs" -mindepth 1 -maxdepth 1 -type d -name '*fastwam*' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr \
  | head -n 5

RUN="$(find "$REPO/logs" -mindepth 1 -maxdepth 1 -type d -name '*fastwam*' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-)"
echo "LATEST_RUN=$RUN"

if test -n "$RUN" && test -d "$RUN"; then
  echo '=== COMMAND ==='
  cat "$RUN/command.txt" 2>/dev/null || true
  echo '=== LOG_TAIL ==='
  tail -n 260 "$RUN/run_embodiment.log" 2>/dev/null || true
  echo '=== RESOURCE_PEAK ==='
  cat "$RUN/resource_monitor/peak.txt" 2>/dev/null || true
  echo '=== CHECKPOINTS ==='
  find "$RUN" -maxdepth 5 -type d -name 'global_step_*' -print 2>/dev/null | sort || true
fi
