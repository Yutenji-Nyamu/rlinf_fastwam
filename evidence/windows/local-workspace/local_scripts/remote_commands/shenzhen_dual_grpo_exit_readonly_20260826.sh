#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
for label_run in \
  "control:$ROOT/grpo-control-formal100-2gpu64x4-b1024-eval5-phys45-v1" \
  "dvac:$ROOT/dvac-global-z-w0to2-formal100-2gpu64x4-b1024-eval5-phys67-v1"
do
  label=${label_run%%:*}
  run=${label_run#*:}
  echo "=== $label timestamps ==="
  stat -c '%y %s %n' "$run/runtime/driver.log" "$run/runtime/exit_code.txt" "$run/runtime/resource.csv"
  echo "=== $label error context ==="
  grep -aEn -B 4 -A 12 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)|RayTaskError|ActorDiedError|SIGTERM|SIGKILL|ConnectionError|RpcError|GCS' "$run/runtime/driver.log" | tail -n 100 || true
  echo "=== $label final tail ==="
  tail -n 80 "$run/runtime/driver.log"
done

echo '=== ray/system ==='
ps -eo pid,user,lstart,etime,cmd | grep -E 'raylet|gcs_server' | grep -v grep || true
journalctl -k --since '2026-08-26 12:00:00' --no-pager 2>/dev/null | grep -Ei 'oom|killed process|nvrm|xid' | tail -n 40 || true
