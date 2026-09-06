#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs
CONTROL=$ROOT/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
PRISM=$ROOT/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389

TZ=Asia/Shanghai date '+snapshot=%F %T %Z'
printf 'action_head=%s\naction_status=' "$(git -C "$WT" rev-parse HEAD)"
git -C "$WT" status --short

for item in "control:$CONTROL" "prism:$PRISM"; do
  label=${item%%:*}; run=${item#*:}
  pid=$(cat "$run/runtime/wrapper.pid")
  pgid=$(cat "$run/runtime/owned.pgid")
  kill -0 "$pid" 2>/dev/null && alive=yes || alive=no
  latest=$(tr '\r' '\n' < "$run/runtime/driver.log" | grep -aE 'Global Step: [0-9]+/[0-9]+' | tail -n1 || true)
  progress=$(tr '\r' '\n' < "$run/runtime/driver.log" | grep -aE 'Generating Rollout Epochs:|Evaluating Rollout Epochs:|Saving checkpoint' | tail -n1 || true)
  fatal=$(grep -Eic 'Traceback|out of memory|CUDA error|RayActorError|worker died|ErrorInitializationFailed|NCCL.*error' "$run/runtime/driver.log" || true)
  printf '%s wrapper_pid=%s owned_pgid=%s alive=%s fatal=%s\n' "$label" "$pid" "$pgid" "$alive" "$fatal"
  printf '%s latest=%s\n%s progress=%s\n' "$label" "$latest" "$label" "$progress"
  printf '%s runtime_identity_files:\n' "$label"
  find "$run/runtime" -maxdepth 1 -type f \( -name '*namespace*' -o -name '*job*' -o -name '*pid*' -o -name 'launch_manifest.txt' \) -printf '%f=' -exec sh -c 'tr "\n" " " < "$1"; echo' sh {} \; | sort
done

echo 'ray_namespaces:'
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/python" - <<'PY'
import collections
import os
import ray
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_action_adv_preflight", logging_level="ERROR")
rows = ray.util.list_named_actors(all_namespaces=True)
groups = collections.defaultdict(list)
for row in rows:
    ns = str(row.get("namespace", ""))
    if ns.startswith("RLinf"):
        groups[ns].append(row.get("name"))
for ns in sorted(groups):
    print(f"namespace={ns} actors={len(groups[ns])} names={sorted(groups[ns])}")
ray.shutdown()
PY

echo 'gpu_processes:'
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi -i 4,5,6,7 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
awk '/^MemAvailable:/ {print "host_mem_available_kib=" $2}' /proc/meminfo
df -h /data | tail -n1
