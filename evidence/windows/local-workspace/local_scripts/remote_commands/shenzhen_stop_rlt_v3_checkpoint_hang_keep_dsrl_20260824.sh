#!/usr/bin/env bash
set -euo pipefail

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ray_address=172.17.0.1:6389
rlt_namespace=RLinf
dsrl_namespace=RLinf_2
rlt_runtime=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v3/runtime
dsrl_runtime=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run

rlt_pgid=$(cat "$rlt_runtime/owned.pgid")
rlt_observer=$(cat "$rlt_runtime/resource_observer.pid")
dsrl_pid=$(cat "$dsrl_runtime/wrapper.pid")
for value in "$rlt_pgid" "$rlt_observer" "$dsrl_pid"; do
  [[ "$value" =~ ^[0-9]+$ ]]
done
kill -0 "$rlt_pgid"
kill -0 "$rlt_observer"
kill -0 "$dsrl_pid"
RAY_ADDRESS="$ray_address" "$venv/bin/ray" status >/dev/null

RAY_ADDRESS="$ray_address" RLT_NAMESPACE="$rlt_namespace" DSRL_NAMESPACE="$dsrl_namespace" \
  "$venv/bin/python" - <<'PY'
import os
import ray

ray.init(
    address=os.environ["RAY_ADDRESS"],
    namespace="codex_pre_stop_rlt_v3",
    logging_level="ERROR",
)
rows = ray.util.list_named_actors(all_namespaces=True)
rlt = sorted(row["name"] for row in rows if row.get("namespace") == os.environ["RLT_NAMESPACE"])
dsrl = sorted(row["name"] for row in rows if row.get("namespace") == os.environ["DSRL_NAMESPACE"])
if len(rlt) != 15:
    raise SystemExit(f"expected 15 RLT named actors, found {len(rlt)}: {rlt}")
if len(dsrl) != 15:
    raise SystemExit(f"expected 15 DSRL named actors, found {len(dsrl)}: {dsrl}")
print(f"pre_stop_rlt_named={len(rlt)} dsrl_named={len(dsrl)}")
ray.shutdown()
PY

printf '%s\n' \
  "authorized_at=$(date --iso-8601=seconds)" \
  'reason=RLT v3 irrecoverably hung in Step25 pre-real-update FSDP/DCP checkpoint; user authorized minimal A/B diagnosis' \
  "owned_pgid=$rlt_pgid" \
  "namespace=$rlt_namespace" \
  "preserved_dsrl_wrapper=$dsrl_pid" \
  "preserved_dsrl_namespace=$dsrl_namespace" \
  > "$rlt_runtime/authorized_stop_for_checkpoint_ab.txt"

kill -TERM -- "-$rlt_pgid"
for _ in $(seq 1 90); do
  if ! kill -0 "$rlt_pgid" 2>/dev/null; then
    break
  fi
  sleep 2
done
if kill -0 "$rlt_pgid" 2>/dev/null; then
  printf '%s\n' "RLT owned process group $rlt_pgid did not stop after 180 seconds" >&2
  exit 3
fi

if kill -0 "$rlt_observer" 2>/dev/null; then
  kill -TERM -- "-$rlt_observer" 2>/dev/null || kill -TERM "$rlt_observer"
fi

RAY_ADDRESS="$ray_address" RLT_NAMESPACE="$rlt_namespace" DSRL_NAMESPACE="$dsrl_namespace" \
  "$venv/bin/python" - <<'PY'
import os
import time
import ray

rlt_namespace = os.environ["RLT_NAMESPACE"]
dsrl_namespace = os.environ["DSRL_NAMESPACE"]
ray.init(
    address=os.environ["RAY_ADDRESS"],
    namespace="codex_cleanup_rlt_v3",
    logging_level="ERROR",
)

def names(namespace):
    return sorted(
        row["name"]
        for row in ray.util.list_named_actors(all_namespaces=True)
        if row.get("namespace") == namespace
    )

targets = names(rlt_namespace)
print(f"target_namespace={rlt_namespace} live_named_actors={len(targets)}")
for name in targets:
    try:
        ray.kill(ray.get_actor(name, namespace=rlt_namespace), no_restart=True)
        print(f"killed={name}")
    except ValueError:
        print(f"already_gone={name}")

for _ in range(90):
    remaining = names(rlt_namespace)
    if not remaining:
        break
    time.sleep(1)
else:
    raise SystemExit(f"RLT namespace cleanup incomplete: {remaining}")

survivors = names(dsrl_namespace)
if len(survivors) != 15:
    raise SystemExit(f"DSRL namespace changed during RLT cleanup: {len(survivors)} {survivors}")
print(f"namespace_cleanup=complete dsrl_named_actors={len(survivors)}")
ray.shutdown()
PY

kill -0 "$dsrl_pid"
for pid in 371579 371581 371582 371584 371587 371591; do
  kill -0 "$pid"
done

for _ in $(seq 1 60); do
  if ! nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits \
      | grep -Eq '^[[:space:]]*[0-9]+'; then
    break
  fi
  sleep 2
done
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
  printf '%s\n' 'RLT GPU 4-5 processes remain after namespace cleanup' >&2
  exit 4
fi

printf '%s\n' \
  "cleaned_at=$(date --iso-8601=seconds)" \
  "rlt_namespace=$rlt_namespace" \
  "dsrl_namespace=$dsrl_namespace" \
  'rlt_checkpoint_preserved=incomplete global_step_25; no deletion' \
  'shared_ray=preserved' \
  'dsrl=preserved' \
  > "$rlt_runtime/rlt_v3_namespace_cleanup.txt"

printf 'rlt_pgid=%s state=stopped\ndsrl_pid=%s state=alive\n' "$rlt_pgid" "$dsrl_pid"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
printf '%s\n' 'SZ_RLT_V3_STOPPED_EXACT_NAMESPACE_DSRL_PRESERVED'
