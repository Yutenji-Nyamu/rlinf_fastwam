#!/usr/bin/env bash
set -euo pipefail

for pid in 3883547 3883549 399688 399691; do
  [[ -r /proc/$pid/environ ]] || continue
  echo "pid=$pid"
  tr '\0' '\n' < /proc/$pid/environ \
    | grep -E '^(RANK|LOCAL_RANK|WORLD_SIZE|LOCAL_WORLD_SIZE|MASTER_ADDR|MASTER_PORT|RAY_JOB_ID|RAY_NAMESPACE|NCCL_[A-Z_]+)=' \
    | sort || true
done
