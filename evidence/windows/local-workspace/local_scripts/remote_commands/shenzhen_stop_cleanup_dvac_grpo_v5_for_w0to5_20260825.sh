#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
TARGET_NAMESPACE=RLinf
TARGET_JOB=20000000

pid=$(cat "$RUN/runtime/wrapper.pid")
pgid=$(cat "$RUN/runtime/owned.pgid")
test "$pid" = "$pgid"
test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
ps -o args= -p "$pid" | grep -F "$RUN/runtime/wrapper.sh" >/dev/null

target_gpu=0
while read -r process_pid; do
  [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
  job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
  if [[ "$job" == "$TARGET_JOB" ]]; then
    test "$(ps -o user= -p "$process_pid" | xargs)" = chenyiteng
    target_gpu=$((target_gpu + 1))
  fi
done < <(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
test "$target_gpu" -eq 12

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

observer=$(cat "$RUN/runtime/observer.pid" 2>/dev/null || true)
if [[ -n "$observer" ]] && kill -0 "$observer" 2>/dev/null; then
  kill -TERM -- "-$observer" 2>/dev/null || kill -TERM "$observer" 2>/dev/null || true
fi

RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$TARGET_NAMESPACE" "$VENV/bin/python" - <<'PY'
import os
import time
import ray

target = os.environ["TARGET_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_stop_dvac_v5_for_w0to5", logging_level="ERROR")

def names():
    return sorted(
        row["name"] for row in ray.util.list_named_actors(all_namespaces=True)
        if row.get("namespace") == target
    )

targets = names()
print(f"target_namespace={target} actors_before={len(targets)} names={targets}")
manager_names = {"CollectiveManager", "DeviceLockManager", "NodeManager", "PortLockManager", "WorkerManager"}
for name in sorted(targets, key=lambda item: (item in manager_names, item)):
    try:
        ray.kill(ray.get_actor(name, namespace=target), no_restart=True)
        print(f"killed={name}")
    except ValueError:
        print(f"already_gone={name}")
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
  while read -r process_pid; do
    [[ -n "$process_pid" && -r "/proc/$process_pid/environ" ]] || continue
    job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    [[ "$job" == "$TARGET_JOB" ]] && target_gpu=$((target_gpu + 1))
  done < <(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
  (( target_gpu == 0 )) && break
  sleep 1
done
test "$target_gpu" -eq 0

cat > "$RUN/runtime/stopped_by_user_for_w0to5.txt" <<EOF
stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)
reason=user requested closeout of DVAC weights[0,2] before fresh weights[0,5] comparison
last_complete_step=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$RUN/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
ray_job_id=$TARGET_JOB
namespace=$TARGET_NAMESPACE
cleanup=exact owned PGID and exact named actors; shared Ray unchanged
EOF

echo '=== post-stop ==='
cat "$RUN/runtime/stopped_by_user_for_w0to5.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,3p'
echo 'SZ_DVAC_GRPO_V5_STOP_FOR_W0TO5_OK'
