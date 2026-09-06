#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
TARGET_NAMESPACE=RLinf
TARGET_JOB=1c000000

pid=$(<"$RUN/runtime/wrapper.pid")
pgid=$(<"$RUN/runtime/owned.pgid")
test "$pid" = "$pgid"
test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
ps -o args= -p "$pid" | grep -F "$RUN/runtime/wrapper.sh" >/dev/null

kill -TERM -- "-$pgid"
for _ in $(seq 1 60); do
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 1
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL -- "-$pgid"
  sleep 2
fi
! kill -0 "$pid" 2>/dev/null

RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$TARGET_NAMESPACE" "$VENV/bin/python" - <<'PY'
import os
import time

import ray

target = os.environ["TARGET_NAMESPACE"]
ray.init(
    address=os.environ["RAY_ADDRESS"],
    namespace="codex_cleanup_wrong_budget_dvac_v3",
    logging_level="ERROR",
)

def names():
    return sorted(
        row["name"]
        for row in ray.util.list_named_actors(all_namespaces=True)
        if row.get("namespace") == target
    )

targets = names()
print(f"target_namespace={target} actors_before={len(targets)} names={targets}")
manager_names = {"CollectiveManager", "NodeManager", "PortLockManager", "WorkerManager"}
ordered = sorted(targets, key=lambda name: (name in manager_names, name))
handles = [(name, ray.get_actor(name, namespace=target)) for name in ordered]
for name, handle in handles:
    ray.kill(handle, no_restart=True)
    print(f"killed={name}")

for _ in range(120):
    remaining = names()
    if not remaining:
        break
    time.sleep(1)
else:
    raise SystemExit(f"namespace cleanup incomplete: {remaining}")
print(f"target_namespace={target} actors_after=0")
ray.shutdown()
PY

for _ in $(seq 1 120); do
  target_gpu=0
  while read -r gpu_pid; do
    [[ -n "$gpu_pid" && -r "/proc/$gpu_pid/environ" ]] || continue
    job=$(tr '\0' '\n' < "/proc/$gpu_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    [[ "$job" == "$TARGET_JOB" ]] && target_gpu=$((target_gpu + 1))
  done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  if (( target_gpu == 0 )); then break; fi
  sleep 1
done
test "$target_gpu" -eq 0

cat > "$RUN/runtime/stopped_wrong_comparison_budget.txt" <<EOF
stopped_at=$(date --iso-8601=seconds)
reason=not a single-variable comparison to Shenzhen GRPO baseline
wrong_budget=32 train env x 8 rollout epochs; 256 trajectories; B512
target_budget=128 train env x 4 rollout epochs; 512 trajectories; B2048
ray_job_id=$TARGET_JOB
namespace=$TARGET_NAMESPACE
cleanup=exact named actors killed with no_restart; shared Ray unchanged
EOF

echo '=== post-stop GPUs ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,2p'
echo "target_job_gpu_processes=$target_gpu"
echo 'SZ_WRONG_BUDGET_DVAC_GRPO_V3_STOP_CLEAN_OK'

