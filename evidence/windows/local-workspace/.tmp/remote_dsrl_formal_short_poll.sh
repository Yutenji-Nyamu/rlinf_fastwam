set -euo pipefail

RUN_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
PID=$(cat "$RUN_ROOT/formal.pid")

echo "TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
kill -0 "$PID" 2>/dev/null && echo "DRIVER_ALIVE=1" || echo "DRIVER_ALIVE=0"
echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
echo "PROCESSES"
pgrep -c -f '^ray::EnvWorker' | awk '{print "ENV_WORKERS=" $1}' || true
pgrep -c -f '^ray::EmbodiedSACFSDPPolicy' | awk '{print "ACTOR_WORKERS=" $1}' || true
pgrep -c -f '^ray::MultiStepRolloutWorker' | awk '{print "ROLLOUT_WORKERS=" $1}' || true
echo "MEMORY"
grep -E '^(anon|file|inactive_file|active_file) ' /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.events
cat /sys/fs/cgroup/memory.pressure
echo "PEAK"
grep -E '^(updated_at|process_alive|current_ram_mb|peak_ram_mb|current_gpu[01]_mb|peak_gpu[01]_mb|current_env_rss_mb|current_actor_rss_mb|current_rollout_rss_mb|peak_env_rss_mb|peak_actor_rss_mb|peak_rollout_rss_mb|cgroup_oom|cgroup_oom_kill)=' \
  "$RUN_ROOT/resource_monitor/peak.txt" || true
echo "PROGRESS"
grep -E 'rollout_epoch|Rollout epoch|rollout .*took|global_new_transitions|planned_optimizer_updates|run_training|global_step|success_once|Saved|checkpoint|Traceback|ERROR|Exception|CUDA out of memory|OutOfMemory|NaN|Inf' \
  "$RUN_ROOT/formal_driver.log" | tail -n 80 || true
echo "LAST_LOG_TIME=$(stat -c '%y' "$RUN_ROOT/formal_driver.log")"
echo "LOG_BYTES=$(stat -c '%s' "$RUN_ROOT/formal_driver.log")"
echo "RUN_BYTES=$(du -sb "$RUN_ROOT" | awk '{print $1}')"
df -BG --output=avail /root/autodl-tmp | tail -n 1 | awk '{print "DISK_FREE=" $1}'
