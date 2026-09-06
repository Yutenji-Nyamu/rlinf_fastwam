#!/usr/bin/env bash
set -euo pipefail

physical_gpu="$1"
node_ip="$2"
gcs_port="$3"
port_base="$4"
worker_min="$5"
worker_max="$6"
runtime_root="$7"

venv=/root/autodl-tmp/RLinf/.venv
ray_temp="$runtime_root/ray_head"
system_temp="$runtime_root/system_tmp"
spill="$runtime_root/object_spill"
mkdir -p "$ray_temp" "$system_temp" "$spill"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export RAY_TMPDIR="$ray_temp"
export TMPDIR="$system_temp"
unset RAY_ADDRESS http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

# One half of the historical Ray CPU declaration and a 24-GiB Plasma store.
# The previous two-GPU run declared 36 CPUs and a 71.6-GiB store but live
# `ray memory --stats-only` reported 0 MiB at the sampled idle boundary.
exec "$venv/bin/ray" start --head --block \
  --node-ip-address="$node_ip" \
  --port="$gcs_port" \
  --dashboard-port="$((port_base + 1))" \
  --dashboard-agent-listen-port="$((port_base + 2))" \
  --dashboard-agent-grpc-port="$((port_base + 3))" \
  --runtime-env-agent-port="$((port_base + 4))" \
  --node-manager-port="$((port_base + 5))" \
  --object-manager-port="$((port_base + 6))" \
  --ray-client-server-port="$((port_base + 7))" \
  --metrics-export-port="$((port_base + 8))" \
  --min-worker-port="$worker_min" \
  --max-worker-port="$worker_max" \
  --num-cpus=18 \
  --num-gpus=1 \
  --object-store-memory=25769803776 \
  --object-spilling-directory="$spill" \
  --temp-dir="$ray_temp" \
  --include-dashboard=false \
  --disable-usage-stats \
  --log-style=record
