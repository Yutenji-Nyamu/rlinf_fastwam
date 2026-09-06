#!/usr/bin/env bash
set -euo pipefail
PACKET=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload
SCRIPT="$PACKET/reloadcheck_worker.sh"
test -f "$SCRIPT"
test ! -e "$PACKET/reloadcheck_worker.pid"
chmod 700 "$SCRIPT"
bash -n "$SCRIPT"
nohup setsid bash "$SCRIPT" > "$PACKET/reloadcheck_worker.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PACKET/reloadcheck_worker.pid"
sleep 12
kill -0 "$pid"
printf 'reloadcheck_pid=%s\nFASTWAM_RELOADCHECK_LAUNCHED\n' "$pid"
