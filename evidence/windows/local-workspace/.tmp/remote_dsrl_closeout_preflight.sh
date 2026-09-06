set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
CKPT_ROOT=$RUN/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1/checkpoints

echo "TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "USER=$(id -un)"
echo "HOST=$(hostname)"
echo "BRANCH=$(git -C "$REPO" branch --show-current)"
echo "HEAD=$(git -C "$REPO" rev-parse HEAD)"
echo "UPSTREAM=$(git -C "$REPO" rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git -C "$REPO" status --short
echo "STATUS_END"

echo "PID_FILES_BEGIN"
for file in "$RUN"/*.pid
do
  test -e "$file" || continue
  pid=$(cat "$file")
  if kill -0 "$pid" 2>/dev/null
  then
    state=alive
  else
    state=dead
  fi
  echo "$(basename "$file")=$pid:$state"
done
echo "PID_FILES_END"

driver=$(cat "$RUN/formal.pid")
echo "PROCESS_TREE_BEGIN"
ps -eo pid,ppid,stat,etimes,rss,args --sort=pid |
  awk -v driver="$driver" '
    $1 == driver ||
    $2 == driver ||
    /EmbodiedSACFSDPPolicy|EnvWorker|RolloutWorker|ray::|raylet|gcs_server/ {
      print
    }
  ' | tail -n 120
echo "PROCESS_TREE_END"

echo "LATEST_STEPS_BEGIN"
grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 8
echo "LATEST_STEPS_END"
echo "LOG_TAIL_BEGIN"
tail -n 100 "$RUN/formal_driver.log"
echo "LOG_TAIL_END"

echo "CHECKPOINTS_BEGIN"
if test -d "$CKPT_ROOT"
then
  for checkpoint in "$CKPT_ROOT"/global_step_*
  do
    test -d "$checkpoint" || continue
    bytes=$(du -sb "$checkpoint" | awk '{print $1}')
    files=$(find "$checkpoint" -type f | wc -l)
    newest=$(find "$checkpoint" -type f -printf '%T@ %TY-%Tm-%TdT%TH:%TM:%TS %p\n' |
      sort -n | tail -n 1)
    temp_count=$(find "$checkpoint" \( -name '*.tmp' -o -name '.metadata.tmp' \) |
      wc -l)
    echo "$(basename "$checkpoint") bytes=$bytes files=$files temp=$temp_count newest=$newest"
  done
fi
echo "CHECKPOINTS_END"

echo "RECENT_WRITES_BEGIN"
find "$RUN" -type f -mmin -5 -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' |
  sort | tail -n 80
echo "RECENT_WRITES_END"

echo "GPU_BEGIN"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
echo "GPU_END"

echo "CGROUP_BEGIN"
cat /sys/fs/cgroup/memory.current
grep -E '^(anon|file|inactive_file|active_file) ' /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.events
cat /sys/fs/cgroup/memory.pressure
echo "CGROUP_END"

echo "DISK_BEGIN"
df -h /root/autodl-tmp
du -sh "$RUN"
du -sh "$RUN"/resource_monitor 2>/dev/null || true
du -sh "$RUN"/tensorboard 2>/dev/null || true
du -sh "$CKPT_ROOT" 2>/dev/null || true
echo "DISK_END"

echo "RUN_FILES_BEGIN"
find "$RUN" -maxdepth 2 -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %P\n' |
  sort -k3
echo "RUN_FILES_END"
