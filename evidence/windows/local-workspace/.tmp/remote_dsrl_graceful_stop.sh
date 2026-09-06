set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=$REPO/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1
CKPT=$RUN/robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1/checkpoints/global_step_195

cd "$REPO"
test "$(git branch --show-current)" = codex/dsrl-pi0-robotwin
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{upstream}')"
test -z "$(git status --porcelain)"

driver=$(cat "$RUN/formal.pid")
kill -0 "$driver"
test -d "$CKPT"
test "$(find "$CKPT" -type f | wc -l)" -eq 11
test -z "$(find "$CKPT" \( -name '*.tmp' -o -name '.metadata.tmp' \))"

collect_descendants() {
  local parent=$1
  local child
  while read -r child
  do
    test -n "$child" || continue
    echo "$child"
    collect_descendants "$child"
  done < <(pgrep -P "$parent" || true)
}

descendants=$(collect_descendants "$driver" | sort -n -u)
requested_at=$(date '+%Y-%m-%d %H:%M:%S %Z')
latest_before=$(grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1)

printf '%s\n' "$requested_at" > "$RUN/formal_stop_requested_at.txt"
printf '%s\n' "$latest_before" > "$RUN/formal_stop_last_completed_step.txt"
printf '%s\n' 'global_step_195' > "$RUN/formal_stop_resume_checkpoint.txt"

echo "STOP_REQUESTED_AT=$requested_at"
echo "DRIVER=$driver"
echo "LAST_COMPLETED_BEFORE=$latest_before"
echo "DESCENDANTS_BEGIN"
printf '%s\n' "$descendants"
echo "DESCENDANTS_END"

kill -TERM "$driver"
for _ in $(seq 1 45)
do
  if ! kill -0 "$driver" 2>/dev/null
  then
    break
  fi
  sleep 1
done

if kill -0 "$driver" 2>/dev/null
then
  echo "DRIVER_TERM_TIMEOUT=1"
  kill -INT "$driver"
  for _ in $(seq 1 30)
  do
    if ! kill -0 "$driver" 2>/dev/null
    then
      break
    fi
    sleep 1
  done
fi

if kill -0 "$driver" 2>/dev/null
then
  echo "DRIVER_STILL_ALIVE=1"
  exit 2
fi
echo "DRIVER_EXITED=1"

for pid in $descendants
do
  if kill -0 "$pid" 2>/dev/null
  then
    kill -TERM "$pid" 2>/dev/null || true
  fi
done

for pid_file in \
  "$RUN/resource_monitor/monitor.pid" \
  "$RUN/resource_monitor/cgroup_detail_monitor.pid"
do
  test -f "$pid_file" || continue
  monitor_pid=$(cat "$pid_file")
  if kill -0 "$monitor_pid" 2>/dev/null
  then
    kill -TERM "$monitor_pid" 2>/dev/null || true
  fi
done
sleep 4

alive_descendants=0
for pid in $descendants
do
  if kill -0 "$pid" 2>/dev/null
  then
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    case "$state" in
      Z*) ;;
      *) echo "DESCENDANT_STILL_ALIVE=$pid:$state"; alive_descendants=1 ;;
    esac
  fi
done

alive_monitors=0
for pid_file in \
  "$RUN/resource_monitor/monitor.pid" \
  "$RUN/resource_monitor/cgroup_detail_monitor.pid"
do
  test -f "$pid_file" || continue
  monitor_pid=$(cat "$pid_file")
  if kill -0 "$monitor_pid" 2>/dev/null
  then
    echo "MONITOR_STILL_ALIVE=$monitor_pid"
    alive_monitors=1
  fi
done

completed_at=$(date '+%Y-%m-%d %H:%M:%S %Z')
printf '%s\n' "$completed_at" > "$RUN/formal_stop_completed_at.txt"
echo "STOP_COMPLETED_AT=$completed_at"
echo "ALIVE_DESCENDANTS=$alive_descendants"
echo "ALIVE_MONITORS=$alive_monitors"
echo "CHECKPOINT_BYTES=$(du -sb "$CKPT" | awk '{print $1}')"
echo "CHECKPOINT_FILES=$(find "$CKPT" -type f | wc -l)"
echo "CHECKPOINT_TEMP=$(find "$CKPT" \( -name '*.tmp' -o -name '.metadata.tmp' \) | wc -l)"
echo "LAST_LOG_LINES_BEGIN"
tail -n 80 "$RUN/formal_driver.log"
echo "LAST_LOG_LINES_END"
echo "GPU_BEGIN"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
echo "GPU_END"
echo "CGROUP_BEGIN"
cat /sys/fs/cgroup/memory.current
grep -E '^(anon|file|inactive_file|active_file) ' /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.events
echo "CGROUP_END"

test "$alive_descendants" -eq 0
test "$alive_monitors" -eq 0
echo "GRACEFUL_STOP=PASS"
