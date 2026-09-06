#!/usr/bin/env bash
set -u

src=/data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1
dst=/data/chenyiteng/results/rlinf-shared-ray/formal-rlt-dsrl-20260823-v1-failed-public-ip
rlt_chain=/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage1-2k-stage2-8env250-20260823-v1
next_tmp=/data/chenyiteng/ray/rlt-dsrl-v2
for path in "$src" "$dst" "$rlt_chain" "$next_tmp"; do
  if [ -e "$path" ]; then printf 'exists=%s\n' "$path"; else printf 'absent=%s\n' "$path"; fi
done
printf '%s\n' '--- user ray processes ---'
pgrep -u "$(id -u)" -af 'raylet|gcs_server|dashboard|monitor.py' || true
printf '%s\n' '--- gpu compute ---'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits || true
printf '%s\n' '--- ports ---'
ss -ltn | grep -E ':(6389|8266)[[:space:]]' || true
printf '%s\n' '--- source start log head ---'
sed -n '1,60p' "$src/ray_start.log" 2>/dev/null || true
