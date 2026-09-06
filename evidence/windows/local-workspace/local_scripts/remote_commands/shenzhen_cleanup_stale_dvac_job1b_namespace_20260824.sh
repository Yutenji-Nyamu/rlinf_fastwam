#!/usr/bin/env bash
set -euo pipefail

VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
OLD_NAMESPACE=RLinf_1
NEW_NAMESPACE=RLinf
OLD_JOB=1b000000
NEW_JOB=1c000000
OLD_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys2367-v2
NEW_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3

old_wrapper=$(<"$OLD_RUN/runtime/wrapper.pid")
new_wrapper=$(<"$NEW_RUN/runtime/wrapper.pid")
! kill -0 "$old_wrapper" 2>/dev/null
kill -0 "$new_wrapper"

RAY_ADDRESS="$RAY_ADDRESS" OLD_NAMESPACE="$OLD_NAMESPACE" NEW_NAMESPACE="$NEW_NAMESPACE" \
OLD_JOB="$OLD_JOB" NEW_JOB="$NEW_JOB" "$VENV/bin/python" - <<'PY'
import os
import time

import ray

old_namespace = os.environ["OLD_NAMESPACE"]
new_namespace = os.environ["NEW_NAMESPACE"]

ray.init(
    address=os.environ["RAY_ADDRESS"],
    namespace="codex_cleanup_stale_dvac_job1b",
    logging_level="ERROR",
)

def names(namespace):
    return sorted(
        row["name"]
        for row in ray.util.list_named_actors(all_namespaces=True)
        if row.get("namespace") == namespace
    )

old_names = names(old_namespace)
new_names_before = names(new_namespace)
if not old_names:
    raise SystemExit(f"no live actors in exact old namespace {old_namespace}")
if not new_names_before:
    raise SystemExit(f"current namespace {new_namespace} unexpectedly empty")

print(f"old_namespace={old_namespace} actors={len(old_names)} names={old_names}")
print(f"new_namespace={new_namespace} actors_before={len(new_names_before)}")

manager_names = {"CollectiveManager", "NodeManager", "PortLockManager", "WorkerManager"}
ordered = sorted(old_names, key=lambda name: (name in manager_names, name))
handles = [(name, ray.get_actor(name, namespace=old_namespace)) for name in ordered]
for name, handle in handles:
    ray.kill(handle, no_restart=True)
    print(f"killed_old_actor={name}")

for _ in range(120):
    remaining = names(old_namespace)
    if not remaining:
        break
    time.sleep(1)
else:
    raise SystemExit(f"old namespace cleanup incomplete: {remaining}")

new_names_after = names(new_namespace)
if new_names_after != new_names_before:
    raise SystemExit(
        "current namespace actor set changed during cleanup: "
        f"before={new_names_before}, after={new_names_after}"
    )
print(f"old_namespace={old_namespace} actors_after=0")
print(f"new_namespace={new_namespace} actors_after={len(new_names_after)} unchanged=yes")
ray.shutdown()
PY

for _ in $(seq 1 120); do
  old_gpu=0
  new_gpu=0
  while read -r pid; do
    [[ -n "$pid" && -r "/proc/$pid/environ" ]] || continue
    job=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    [[ "$job" == "$OLD_JOB" ]] && old_gpu=$((old_gpu + 1))
    [[ "$job" == "$NEW_JOB" ]] && new_gpu=$((new_gpu + 1))
  done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  if (( old_gpu == 0 )); then break; fi
  sleep 1
done

if (( old_gpu != 0 )); then
  echo "old GPU actors remain: $old_gpu" >&2
  exit 31
fi
if (( new_gpu != 12 )); then
  echo "current GPU actor count changed unexpectedly: $new_gpu" >&2
  exit 32
fi
kill -0 "$new_wrapper"

cat > "$OLD_RUN/runtime/stale_ray_actor_cleanup.txt" <<EOF
cleaned_at=$(date --iso-8601=seconds)
ray_job_id=$OLD_JOB
namespace=$OLD_NAMESPACE
scope=stale actors from stopped physical 2,3,6,7 run only
method=ray.kill(no_restart=True) on exact named actors; shared Ray unchanged
current_job_id=$NEW_JOB
current_namespace=$NEW_NAMESPACE
EOF

echo '=== post-clean resources ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,2p'
echo "old_job_gpu_processes=$old_gpu"
echo "current_job_gpu_processes=$new_gpu"
echo "current_wrapper_alive=yes pid=$new_wrapper"
grep -E 'Global Step:|Generating Rollout Epochs:' "$NEW_RUN/runtime/driver.log" | tail -n 5 || true
printf 'current_fatal_matches='; grep -Eci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$NEW_RUN/runtime/driver.log" || true
echo 'SZ_STALE_DVAC_JOB1B_NAMESPACE_CLEANUP_OK'

