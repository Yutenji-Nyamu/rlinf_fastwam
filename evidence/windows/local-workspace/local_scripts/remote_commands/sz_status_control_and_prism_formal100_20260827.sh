#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
PRISM=$ROOT/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389

TZ=Asia/Shanghai date --iso-8601=seconds
for label_run in "control:$CONTROL" "prism:$PRISM"; do
  label=${label_run%%:*}; run=${label_run#*:}
  pid=$(<"$run/runtime/wrapper.pid")
  printf '%s alive=' "$label"
  if kill -0 "$pid" 2>/dev/null; then printf 'yes'; else printf 'no'; fi
  printf ' exit='
  if [[ -f "$run/runtime/exit_code.txt" ]]; then cat "$run/runtime/exit_code.txt"; else echo pending; fi
  printf '%s fatal_matches=' "$label"
  grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$run/runtime/driver.log" 2>/dev/null || true
  grep -aE 'Global Step:|Generating Rollout Epochs:|Creating rollout worker|Creating env worker|Creating actor worker|Loading checkpoint shards' "$run/runtime/driver.log" 2>/dev/null | tail -n 8 || true
done

RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_control_prism_status", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
for ns in sorted({row.get("namespace") for row in rows if str(row.get("namespace", "")).startswith("RLinf")}):
    names = sorted(row["name"] for row in rows if row.get("namespace") == ns)
    print(f"namespace={ns} actor_count={len(names)} names={names}")
ray.shutdown()
PY

echo 'GPU jobs:'
for process_pid in $(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu); do
  [[ -r "/proc/$process_pid/environ" ]] || continue
  job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
  printf 'pid=%s job=%s user=%s cmd=%s\n' "$process_pid" "${job:-none}" "$(ps -o user= -p "$process_pid" | xargs)" "$(ps -o comm= -p "$process_pid" | xargs)"
done
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/^MemAvailable:/ {printf "host_mem_available_gib=%.3f\n", $2/1024/1024}' /proc/meminfo
