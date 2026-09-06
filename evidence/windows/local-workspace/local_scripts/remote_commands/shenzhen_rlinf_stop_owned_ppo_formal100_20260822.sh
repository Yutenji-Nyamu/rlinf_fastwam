#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1
PID_FILE="$RUN/driver.pid"

test -s "$PID_FILE"
pid=$(tr -d '[:space:]' < "$PID_FILE")
case "$pid" in
  ''|*[!0-9]*) printf 'invalid driver pid: %s\n' "$pid" >&2; exit 1 ;;
esac

if ! kill -0 "$pid" 2>/dev/null; then
  printf 'owned PPO driver already exited: pid=%s\n' "$pid"
  exit 0
fi

test "$(stat -c %u "/proc/$pid")" = "$(id -u)"
cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
case "$cmdline" in
  *'/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin/examples/embodiment/train_embodied_agent.py'*) ;;
  *) printf 'refusing unexpected driver cmdline: %s\n' "$cmdline" >&2; exit 1 ;;
esac

pgid=$(ps -o pgid= -p "$pid" | tr -d '[:space:]')
test "$pgid" = "$pid"

printf 'stop_time=%s\n' "$(date --iso-8601=seconds)"
printf 'owned_driver_pid=%s pgid=%s\n' "$pid" "$pgid"
printf 'owned_driver_cmdline=%s\n' "$cmdline"
grep -F 'Global Step' "$RUN/metrics.log" | tail -n 1 || true
printf 'sending SIGINT to owned PPO process group %s\n' "$pgid"
kill -INT -- "-$pgid"

for _ in $(seq 1 60); do
  if ! kill -0 "$pid" 2>/dev/null; then
    printf '%s\n' 'owned PPO driver exited after SIGINT'
    break
  fi
  sleep 2
done

if kill -0 "$pid" 2>/dev/null; then
  printf 'sending SIGTERM to still-alive owned PPO process group %s\n' "$pgid"
  kill -TERM -- "-$pgid"
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
fi

if kill -0 "$pid" 2>/dev/null; then
  printf 'owned PPO driver remains alive; no SIGKILL sent\n' >&2
  exit 1
fi

sleep 5
printf '%s\n' '--- remaining chenyiteng Ray/train processes ---'
ps -u "$(id -un)" -o pid,ppid,pgid,stat,rss,comm,args \
  | grep -E 'train_embodied_agent.py|raylet|gcs_server|ray::' \
  | grep -v grep || true
printf '%s\n' '--- GPU 4-7 compute apps ---'
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,used_memory \
  --format=csv,noheader,nounits || true
printf '%s\n' '--- host memory ---'
awk '/^(MemTotal|MemAvailable):/ {print}' /proc/meminfo
printf '%s\n' 'SZ_PPO_FORMAL100_OWNED_STOP_COMPLETE'
