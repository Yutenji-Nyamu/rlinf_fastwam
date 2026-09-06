#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RAY_ADDRESS=172.17.0.1:6389
TARGET_NAMESPACE=RLinf

pid=$(<"$RUN/runtime/wrapper.pid")
pgid=$(<"$RUN/runtime/owned.pgid")
test "$pid" = "$pgid"
test "$(ps -o user= -p "$pid" | xargs)" = chenyiteng
ps -o args= -p "$pid" | grep -F "$RUN/runtime/wrapper.sh" >/dev/null

mapfile -t gpu_pids < <(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu)
test "${#gpu_pids[@]}" -eq 12
jobs=()
for process_pid in "${gpu_pids[@]}"; do
  test "$(ps -o user= -p "$process_pid" | xargs)" = chenyiteng
  job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
  test -n "$job"
  jobs+=("$job")
done
mapfile -t unique_jobs < <(printf '%s\n' "${jobs[@]}" | sort -u)
test "${#unique_jobs[@]}" -eq 1
TARGET_JOB=${unique_jobs[0]}

RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$TARGET_NAMESPACE" "$VENV/bin/python" - <<'PY'
import os, ray
target = os.environ["TARGET_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_dual_grpo_stop_precheck", logging_level="ERROR")
names = sorted(row["name"] for row in ray.util.list_named_actors(all_namespaces=True) if row.get("namespace") == target)
assert len(names) == 21, names
print(f"stop_precheck namespace={target} actors={len(names)}")
ray.shutdown()
PY

last_complete=$(grep -aoE 'Global Step:[[:space:]]+[0-9]+/100' "$RUN/runtime/driver.log" | tail -n1 | grep -oE '[0-9]+' | head -n1)
kill -TERM -- "-$pgid"
for _ in $(seq 1 60); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL -- "-$pgid"
  sleep 2
fi
! kill -0 "$pid" 2>/dev/null

observer=$(<"$RUN/runtime/observer.pid" 2>/dev/null || true)
if [[ -n "$observer" ]] && kill -0 "$observer" 2>/dev/null; then
  kill -TERM -- "-$observer" 2>/dev/null || kill -TERM "$observer" 2>/dev/null || true
fi

RAY_ADDRESS="$RAY_ADDRESS" TARGET_NAMESPACE="$TARGET_NAMESPACE" "$VENV/bin/python" - <<'PY'
import os, time, ray
target = os.environ["TARGET_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_dual_grpo_stop", logging_level="ERROR")
manager_names = {"CollectiveManager", "DeviceLockManager", "NodeManager", "PortLockManager", "WorkerManager"}
def names():
    return sorted(row["name"] for row in ray.util.list_named_actors(all_namespaces=True) if row.get("namespace") == target)
before = names()
print(f"target_namespace={target} actors_before={len(before)}")
for name in sorted(before, key=lambda item: (item in manager_names, item)):
    try:
        ray.kill(ray.get_actor(name, namespace=target), no_restart=True)
    except ValueError:
        pass
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
  remaining=0
  for process_pid in $(nvidia-smi -i 4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -nu); do
    [[ -r "/proc/$process_pid/environ" ]] || continue
    job=$(tr '\0' '\n' < "/proc/$process_pid/environ" | sed -n 's/^RAY_JOB_ID=//p' | head -n1)
    [[ "$job" == "$TARGET_JOB" ]] && remaining=$((remaining + 1))
  done
  (( remaining == 0 )) && break
  sleep 1
done
test "$remaining" -eq 0

printf '%s\n' \
  "stopped_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" \
  'reason=user requested replacement by paired 2-GPU GRPO control and DVAC [0,2] formal runs' \
  "last_complete_step=$last_complete" \
  "ray_job_id=$TARGET_JOB" \
  "namespace=$TARGET_NAMESPACE" \
  'cleanup=exact owned PGID and exact namespace actors; shared Ray unchanged' \
  > "$RUN/runtime/stopped_by_user_for_dual_2gpu.txt"
RAY_ADDRESS="$RAY_ADDRESS" "$VENV/bin/ray" status >/dev/null
cat "$RUN/runtime/stopped_by_user_for_dual_2gpu.txt"
echo SZ_W0TO5_STOPPED_FOR_DUAL_GRPO_OK

