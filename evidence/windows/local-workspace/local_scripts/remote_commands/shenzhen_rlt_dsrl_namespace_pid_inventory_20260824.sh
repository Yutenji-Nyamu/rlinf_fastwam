#!/usr/bin/env bash
set -euo pipefail

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ray_address=172.17.0.1:6389

RAY_ADDRESS="$ray_address" "$venv/bin/python" - <<'PY'
import os
import ray
from ray.util.state import list_actors

ray.init(
    address=os.environ["RAY_ADDRESS"],
    namespace="codex_readonly_namespace_inventory",
    logging_level="ERROR",
)

print("--- named actors by namespace ---")
rows = ray.util.list_named_actors(all_namespaces=True)
for row in sorted(rows, key=lambda item: (item.get("namespace", ""), item.get("name", ""))):
    print(f"{row.get('namespace')}|{row.get('name')}")

print("--- live actor pid/class/job/namespace ---")
for actor in list_actors(detail=True, limit=1000):
    if actor.get("state") != "ALIVE":
        continue
    class_name = actor.get("class_name", "")
    if not any(token in class_name for token in ("RLT", "SAC", "Rollout", "EnvWorker")):
        continue
    print(
        f"{actor.get('pid')}|{actor.get('ray_namespace')}|{actor.get('job_id')}|"
        f"{actor.get('name')}|{class_name}"
    )
ray.shutdown()
PY
