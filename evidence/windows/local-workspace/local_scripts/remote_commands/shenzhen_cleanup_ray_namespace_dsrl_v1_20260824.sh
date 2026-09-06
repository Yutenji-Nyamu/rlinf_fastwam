#!/usr/bin/env bash
set -euo pipefail

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ray_address=172.17.0.1:6389
target_namespace=RLinf_1
old_log=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v1/run
new_log=/data/chenyiteng/results/rlinf-current-dsrl/formal-dsrl-pi0-robotwin-2gpu4env-200c-20260824-v2/run
rlt_runtime=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260824-v2/runtime

old_pid=$(cat "$old_log/owned.pgid")
new_pid=$(cat "$new_log/wrapper.pid")
rlt_pid=$(cat "$rlt_runtime/wrapper.pid")
! kill -0 "$old_pid" 2>/dev/null
kill -0 "$new_pid"
kill -0 "$rlt_pid"

RAY_ADDRESS="$ray_address" TARGET_NAMESPACE="$target_namespace" "$venv/bin/python" - <<'PY'
import os
import time
import ray

target = os.environ["TARGET_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="codex_cleanup_dsrl_v1", logging_level="ERROR")

def target_names():
    rows = ray.util.list_named_actors(all_namespaces=True)
    return sorted(
        row["name"]
        for row in rows
        if row.get("namespace") == target
    )

names = target_names()
if not names:
    raise SystemExit(f"no live named actors found in expected namespace {target}")
print(f"target_namespace={target} live_named_actors={len(names)}")
for name in names:
    handle = ray.get_actor(name, namespace=target)
    ray.kill(handle, no_restart=True)
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

kill -0 "$new_pid"
kill -0 "$rlt_pid"
printf '%s\n' \
  "cleaned_at=$(date --iso-8601=seconds)" \
  "namespace=$target_namespace" \
  'scope=stale actors from stopped DSRL v1 only' \
  > "$old_log/stale_namespace_cleanup.txt"
printf 'dsrl_v2_pid=%s state=alive\nrlt_pid=%s state=alive\n' "$new_pid" "$rlt_pid"
printf '%s\n' 'SZ_DSRL_V1_NAMESPACE_CLEANED'
