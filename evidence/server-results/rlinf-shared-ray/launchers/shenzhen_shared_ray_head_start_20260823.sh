#!/usr/bin/env bash
set -euo pipefail

venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
runtime=/data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1
# Ray's AF_UNIX socket path is capped at 107 bytes; keep its session root short.
ray_tmp=/data/chenyiteng/ray/rlt-dsrl-v1
address=127.0.0.1:6389

test -x "$venv/bin/ray"
test ! -e "$runtime"
if pgrep -u "$(id -u)" -x raylet >/dev/null || pgrep -u "$(id -u)" -x gcs_server >/dev/null; then
  printf '%s\n' 'chenyiteng already has a Ray runtime' >&2
  exit 1
fi

mkdir -p "$runtime" "$ray_tmp"
unset CUDA_VISIBLE_DEVICES
export RLINF_NODE_RANK=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DEDUP_LOGS=0

printf '%q ' \
  "$venv/bin/ray" start --head --port=6389 \
  --dashboard-host=127.0.0.1 --dashboard-port=8266 \
  --temp-dir="$ray_tmp" --disable-usage-stats \
  > "$runtime/command.txt"
printf '\n' >> "$runtime/command.txt"

date --iso-8601=seconds > "$runtime/started_at.txt"
"$venv/bin/ray" start --head --port=6389 \
  --dashboard-host=127.0.0.1 --dashboard-port=8266 \
  --temp-dir="$ray_tmp" --disable-usage-stats \
  > "$runtime/ray_start.log" 2>&1
printf '%s\n' "$address" > "$runtime/address.txt"
RAY_ADDRESS="$address" "$venv/bin/ray" status > "$runtime/ray_status.txt"

printf 'ray_address=%s\n' "$address"
printf 'raylet_count='; pgrep -u "$(id -u)" -x raylet | wc -l
printf 'gcs_count='; pgrep -u "$(id -u)" -x gcs_server | wc -l
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 30 "$runtime/ray_start.log"
printf '%s\n' 'SZ_SHARED_RAY_HEAD_READY'
