#!/usr/bin/env bash
set -euo pipefail
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389

TZ=Asia/Shanghai date --iso-8601=seconds
pid=$(cat "$RUN/runtime/wrapper.pid")
pgid=$(cat "$RUN/runtime/owned.pgid")
test "$pid" = "$pgid"
kill -0 "$pid"
printf 'wrapper_pid=%s pgid=%s user=%s\n' "$pid" "$pgid" "$(ps -o user= -p "$pid" | xargs)"
grep -aE 'Global Step:|Generating Rollout Epochs:' "$RUN/runtime/driver.log" | tail -n 12 || true
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$RUN/runtime/driver.log" || true

echo '=== target GPU process job ids ==='
for process_pid in $(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu); do
  [[ -r "/proc/$process_pid/environ" ]] || continue
  job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
  printf 'pid=%s job=%s user=%s\n' "$process_pid" "${job:-none}" "$(ps -o user= -p "$process_pid" | xargs)"
done

RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os
import ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_v5_stop_preflight", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
for row in sorted(rows, key=lambda x: (x.get("namespace", ""), x.get("name", ""))):
    print(f"named_actor namespace={row.get('namespace')} name={row.get('name')}")
ray.shutdown()
PY

nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,3p'
