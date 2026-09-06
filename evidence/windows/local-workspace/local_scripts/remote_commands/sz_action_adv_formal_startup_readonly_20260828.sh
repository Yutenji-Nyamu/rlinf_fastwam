#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
ACTION=$ROOT/dvac-action-adv-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
PRISM=$ROOT/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
TZ=Asia/Shanghai date '+snapshot=%F %T %Z'
for item in "action:$ACTION" "prism:$PRISM"; do
  label=${item%%:*}; run=${item#*:}; pid=$(cat "$run/runtime/wrapper.pid")
  kill -0 "$pid" 2>/dev/null && alive=yes || alive=no
  fatal=$(grep -Eic 'Traceback|out of memory|CUDA error|RayActorError|worker died|ErrorInitializationFailed|NCCL.*error' "$run/runtime/driver.log" || true)
  progress=$(tr '\r' '\n' < "$run/runtime/driver.log" | grep -aE 'Global Step:|Generating Rollout Epochs:|Evaluating Rollout Epochs:|Saving checkpoint|register.*workers|Creating.*Worker' | tail -n2 | tr '\n' ' ' || true)
  printf '%s wrapper_pid=%s alive=%s fatal=%s progress=%s\n' "$label" "$pid" "$alive" "$fatal" "$progress"
done
echo 'ray_namespaces:'
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import collections, os, ray
from ray.util.state import list_actors
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_action_adv_startup", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
groups = collections.defaultdict(list)
for row in rows:
    ns = str(row.get("namespace", ""))
    if ns.startswith("RLinf"):
        groups[ns].append(row.get("name"))
for ns in sorted(groups):
    print(f"namespace={ns} actors={len(groups[ns])}")
for row in list_actors(detail=True, filters=[("state", "=", "ALIVE")]):
    ns = str(getattr(row, "ray_namespace", ""))
    if ns.startswith("RLinf") and getattr(row, "name", None) in {"Actor:0", "Env:0", "Rollout:0"}:
        print(f"actor namespace={ns} name={row.name} pid={row.pid} job_id={row.job_id}")
ray.shutdown()
PY
echo 'gpu_jobs:'
for devices in 4,5 6,7; do
  printf 'gpu%s ' "$devices"
  for pid in $(nvidia-smi -i "$devices" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu); do
    job=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    printf '%s:%s ' "$pid" "$job"
  done
  echo
done
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
awk '/^MemAvailable:/ {print "host_mem_available_kib=" $2}' /proc/meminfo
