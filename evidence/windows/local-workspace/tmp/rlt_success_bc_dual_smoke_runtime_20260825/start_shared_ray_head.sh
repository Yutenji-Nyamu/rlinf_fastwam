#!/usr/bin/env bash
set -euo pipefail

node_ip="$1"
runtime_root="$2"
venv=/root/autodl-tmp/RLinf/.venv
ray_temp=/tmp/ray_shared_50001
system_temp=/tmp/raytmp_shared_50001
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
  --port=50001 \
  --dashboard-port=50002 \
  --dashboard-agent-listen-port=50003 \
  --dashboard-agent-grpc-port=50004 \
  --runtime-env-agent-port=50005 \
  --node-manager-port=50006 \
  --object-manager-port=50007 \
  --ray-client-server-port=50008 \
  --metrics-export-port=50009 \
  --min-worker-port=50100 \
  --max-worker-port=51099 \
  --num-cpus=36 \
  --num-gpus=2 \
  --object-store-memory=51539607552 \
  --object-spilling-directory="$spill" \
  --temp-dir="$ray_temp" \
  --include-dashboard=false \
  --disable-usage-stats \
  --log-style=record
