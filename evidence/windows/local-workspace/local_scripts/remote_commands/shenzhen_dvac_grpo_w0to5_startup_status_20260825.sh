#!/usr/bin/env bash
set -euo pipefail

RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1
pid=$(cat "$RUN/runtime/wrapper.pid")
TZ=Asia/Shanghai date --iso-8601=seconds
kill -0 "$pid"
echo "wrapper_alive=yes pid=$pid"
printf 'fatal_matches='; grep -aEci 'Traceback|OutOfMemory|CUDA out of memory|WorkerCrashed|non[-_ ]?finite|NCCL.*(error|timeout)' "$RUN/runtime/driver.log" || true
grep -aE 'Creating Actors|Creating Environments|Generating Rollout Epochs:|Global Step:|Policy worker initialized|Env worker initialized' "$RUN/runtime/driver.log" | tail -n 30 || true
RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import ray
ray.init(address="172.17.0.1:6389", namespace="RLinf", logging_level="ERROR")
actors = ray.util.list_named_actors(all_namespaces=False)
print(f"named_actors={len(actors)}")
ray.shutdown()
PY
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h | sed -n '1,3p'
echo SZ_DVAC_GRPO_W0TO5_STARTUP_STATUS_OK
