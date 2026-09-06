#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/parity-m10-phys4-v4
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-m10-phys4-v4
test "$(git -C "$WT" rev-parse HEAD)" = 1f4d35c1335d759b9a2e8264e40c136ca08f9ce8
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPU4 is occupied; refusing Sidney parity' >&2
  exit 20
fi
mkdir -p "$RUN/runtime"
monitor() {
  while true; do
    date --iso-8601=seconds
    nvidia-smi -i 4 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    awk '/MemAvailable:/ {print}' /proc/meminfo
    sleep 30
  done
}
monitor > "$RUN/runtime/resources.log" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT
set +e
bash -e "$PACKET/commands.txt" > "$RUN/runtime/wrapper.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$RUN/runtime/exit_code.txt"
exit "$RC"
