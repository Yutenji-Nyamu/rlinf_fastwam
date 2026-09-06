#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PACKET=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/packets/parity-core224-m10-phys4-v7
RUN=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/smokes/parity-core224-m10-phys4-v7
test "$(git -C "$WT" rev-parse HEAD)" = 021fa975fee9db66d01dedd09f203f6a8d4f7877
test -z "$(git -C "$WT" status --porcelain)"
if nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPU4 is occupied; refusing Sidney core parity' >&2
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
