#!/usr/bin/env bash
set -euo pipefail

PACKET=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip
test -x "$PACKET/worker.sh"
test ! -e "$PACKET/worker.pid"

nohup setsid bash "$PACKET/worker.sh" > "$PACKET/worker.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PACKET/worker.pid"
sleep 12
kill -0 "$pid"
printf 'worker_pid=%s\n' "$pid"
sed -n '1,20p' "$PACKET/worker.log"
nvidia-smi -i 2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf 'FASTWAM_SMOKE_LAUNCHED\n'
