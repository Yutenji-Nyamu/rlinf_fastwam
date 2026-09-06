#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
cd "$WT"
printf 'head='; git rev-parse HEAD
printf 'branch='; git branch --show-current
printf '%s\n' 'status_begin'
git status --short
printf '%s\n' 'status_end'
printf 'target_actors='; RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import ray
ray.init(address="172.17.0.1:6389", namespace="RLinf", logging_level="ERROR")
print(len(ray.util.list_named_actors(all_namespaces=False)))
ray.shutdown()
PY
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
