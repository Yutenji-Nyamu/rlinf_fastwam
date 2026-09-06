#!/usr/bin/env bash
set -euo pipefail

node_ip="$1"
runtime_root="$2"
venv=/root/autodl-tmp/RLinf/.venv
ray_temp=/tmp/ray_shared_52001
system_temp=/tmp/raytmp_shared_52001
spill="$runtime_root/object_spill"
mkdir -p "$ray_temp" "$system_temp" "$spill"
printf '%s\n' "$ray_temp" >"$runtime_root/ray_temp_dir.txt"

export CUDA_VISIBLE_DEVICES=0,1
export TORCHINDUCTOR_COMPILE_THREADS=1
export RAY_TMPDIR="$ray_temp"
export TMPDIR="$system_temp"
unset RAY_ADDRESS http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

exec "$venv/bin/ray" start --head --block \
  --node-ip-address="$node_ip" \
  --port=52001 \
  --dashboard-port=52002 \
  --dashboard-agent-listen-port=52003 \
  --dashboard-agent-grpc-port=52004 \
  --runtime-env-agent-port=52005 \
  --node-manager-port=52006 \
  --object-manager-port=52007 \
  --ray-client-server-port=52008 \
  --metrics-export-port=52009 \
  --min-worker-port=52100 \
  --max-worker-port=53099 \
  --num-cpus=36 \
  --num-gpus=2 \
  --object-store-memory=51539607552 \
  --object-spilling-directory="$spill" \
  --temp-dir="$ray_temp" \
  --include-dashboard=false \
  --disable-usage-stats \
  --log-style=record
