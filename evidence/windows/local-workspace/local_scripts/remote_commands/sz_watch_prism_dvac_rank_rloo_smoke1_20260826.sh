#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-smoke1-2gpu64x4-b1024-noeval-phys23-v1
RUNTIME="$RUN/runtime"
PID=$(cat "$RUNTIME/wrapper.pid")

for _ in $(seq 1 75); do
  now=$(TZ=Asia/Shanghai date '+%F %T %Z')
  mem=$(awk '/^MemAvailable:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
  gpu=$(nvidia-smi -i 2,3 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')
  step=$(grep -aoE 'Global Step[^0-9]*[0-9]+' "$RUNTIME/driver.log" 2>/dev/null | tail -n 1 || true)
  bytes=$(stat -c %s "$RUNTIME/driver.log" 2>/dev/null || printf 0)
  if kill -0 "$PID" 2>/dev/null; then
    printf 'WATCH %s alive=1 log_bytes=%s latest="%s" mem_available_gib=%s gpu2_3="%s"\n' \
      "$now" "$bytes" "$step" "$mem" "$gpu"
    sleep 60
    continue
  fi

  rc=$(cat "$RUNTIME/exit_code.txt" 2>/dev/null || printf missing)
  printf 'WATCH_FINAL %s alive=0 exit_code=%s log_bytes=%s latest="%s" mem_available_gib=%s gpu2_3="%s"\n' \
    "$now" "$rc" "$bytes" "$step" "$mem" "$gpu"
  printf '%s\n' '--- filtered terminal evidence ---'
  grep -aiE 'Global Step|prism|trajectory|success|advantage|loss|kl|clip|grad|checkpoint|saving|saved|Traceback|ERROR|Exception|OutOfMemory|OOM|nonfinite|nan|inf' \
    "$RUNTIME/driver.log" | tail -n 160 || true
  printf '%s\n' '--- driver tail ---'
  tail -n 100 "$RUNTIME/driver.log" || true
  printf '%s\n' '--- checkpoints ---'
  find "$RUN/checkpoints" -maxdepth 3 -type f -printf '%p %s\n' 2>/dev/null | sort || true
  printf '%s\n' '--- resource tail ---'
  tail -n 20 "$RUNTIME/resource.csv" || true
  exit 0
done

echo 'WATCH_TIMEOUT wrapper still alive after 75 minutes'
exit 124
