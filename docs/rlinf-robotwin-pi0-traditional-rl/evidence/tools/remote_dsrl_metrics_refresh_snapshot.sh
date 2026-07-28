set -euo pipefail
# Read-only live snapshot used by FORMAL-010.

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1

echo "TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "HOST=$(hostname)"
echo "USER=$(id -un)"

cd "$REPO"
echo "BRANCH=$(git branch --show-current)"
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"

PID=$(cat "$RUN/formal.pid")
kill -0 "$PID"
echo "DRIVER_PID=$PID"
echo "DRIVER_ALIVE=1"
pgrep -c -f '^ray::EmbodiedSACFSDPPolicy' | awk '{print "ACTOR_WORKERS=" $1}'
pgrep -c -f '^ray::MultiStepRolloutWorker' | awk '{print "ROLLOUT_WORKERS=" $1}'
pgrep -c -f '^ray::EnvWorker' | awk '{print "ENV_WORKERS=" $1}'

echo "LAST_GLOBAL_STEP"
grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1
echo "LAST_REPLAY"
grep 'sac/global_resident_transitions=' "$RUN/formal_driver.log" | tail -n 1
echo "LAST_LOG_TIME=$(stat -c '%y' "$RUN/formal_driver.log")"
echo "LOG_BYTES=$(stat -c '%s' "$RUN/formal_driver.log")"

echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
echo "CGROUP_CURRENT=$(cat /sys/fs/cgroup/memory.current)"
echo "CGROUP_MAX=$(cat /sys/fs/cgroup/memory.max)"
grep -E '^(anon|file|inactive_file|active_file) ' /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.events
cat /sys/fs/cgroup/memory.pressure

echo "CHECKPOINTS"
find "$RUN" -type d -path '*/checkpoints/global_step_*' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
echo "CHECKPOINT_SIZES"
find "$RUN" -type d -path '*/checkpoints/global_step_*' -print0 |
  xargs -0 -r du -sh
echo "EVENT_FILES"
find "$RUN/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' -printf '%s %p\n'
echo "RUN_SIZE=$(du -sh "$RUN" | awk '{print $1}')"
df -BG --output=avail /root/autodl-tmp | tail -n 1 | awk '{print "DISK_FREE=" $1}'
sha256sum "$RUN/FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml"
