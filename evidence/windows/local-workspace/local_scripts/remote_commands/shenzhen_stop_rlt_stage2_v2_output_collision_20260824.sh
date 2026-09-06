#!/usr/bin/env bash
set -euo pipefail

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ray_address=172.17.0.1:6389
target_namespace=RLinf
rlt=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2
runtime=$rlt/runtime
stage2_runtime=$rlt/stage2/runtime
dsrl_log=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run

test "$(cat "$rlt/stage1/runtime/exit_code.txt")" = 0
test -s "$rlt/stage1/robotwin_adjust_bottle_rlt_stage1_current_ar_clean50_2k_v1/checkpoints/global_step_2000/actor/model_state_dict/full_weights.pt"
pgid=$(cat "$runtime/owned.pgid")
dsrl_pid=$(cat "$dsrl_log/wrapper.pid")
case "$pgid" in (*[!0-9]*|'') exit 2;; esac
case "$dsrl_pid" in (*[!0-9]*|'') exit 2;; esac
kill -0 "$pgid"
kill -0 "$dsrl_pid"

printf '%s\n' \
  'reason=auto Stage2 inherited env train/eval task_config.save_path=./data under external Ray worker cwd /home/chenyiteng' \
  "stopped_at=$(date --iso-8601=seconds)" \
  'scope=RLT v2 chain Stage2 only after successful Stage1 global_step_2000; DSRL v2 and shared Ray preserved' \
  > "$stage2_runtime/manual_stop.txt"

kill -TERM -- "-$pgid"
for _ in $(seq 1 90); do
  if ! kill -0 "$pgid" 2>/dev/null; then
    break
  fi
  sleep 2
done
if kill -0 "$pgid" 2>/dev/null; then
  printf '%s\n' "RLT owned process group $pgid did not stop after 180 seconds" >&2
  exit 3
fi

observer_pid=$(cat "$runtime/resource_observer.pid" 2>/dev/null || true)
if [[ "$observer_pid" =~ ^[0-9]+$ ]] && kill -0 "$observer_pid" 2>/dev/null; then
  kill -TERM -- "-$observer_pid" 2>/dev/null || kill -TERM "$observer_pid" 2>/dev/null || true
fi

RAY_ADDRESS="$ray_address" TARGET_NAMESPACE="$target_namespace" "$venv/bin/python" - <<'PY'
import os
import time
import ray

target = os.environ["TARGET_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_cleanup_rlt_v2", logging_level="ERROR")

def target_names():
    rows = ray.util.list_named_actors(all_namespaces=True)
    return sorted(row["name"] for row in rows if row.get("namespace") == target)

names = target_names()
if not names:
    raise SystemExit(f"no live named actors found in expected namespace {target}")
print(f"target_namespace={target} live_named_actors={len(names)}")
for name in names:
    ray.kill(ray.get_actor(name, namespace=target), no_restart=True)
    print(f"killed={name}")

for _ in range(60):
    remaining = target_names()
    if not remaining:
        break
    time.sleep(1)
else:
    raise SystemExit(f"namespace cleanup incomplete: {remaining}")
ray.shutdown()
print("namespace_cleanup=complete")
PY

kill -0 "$dsrl_pid"
printf '%s\n' \
  "cleaned_at=$(date --iso-8601=seconds)" \
  "namespace=$target_namespace" \
  'scope=actors from stopped RLT v2 Stage2 only' \
  > "$stage2_runtime/stale_namespace_cleanup.txt"
printf 'rlt_v2_pgid=%s state=stopped\ndsrl_v2_pid=%s state=alive\n' "$pgid" "$dsrl_pid"
printf '%s\n' 'SZ_RLT_STAGE2_V2_STOPPED_AND_NAMESPACE_CLEANED'
