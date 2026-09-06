#!/usr/bin/env bash
set -u

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
RUN=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v1
PACKET=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v1
RELOAD="${RUN}-reloadcheck"

printf 'head=%s\n' "$(git -C "$WT" rev-parse HEAD)"
printf 'dirty=%q\n' "$(git -C "$WT" status --porcelain)"
for path in \
  /data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt \
  /data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json \
  /data/chenyiteng/models/fastwam/diffsynth \
  /home/chenyiteng/cache/fastwam-7faa/modelscope \
  "$RUN" "$RELOAD" "$PACKET"; do
  if test -e "$path"; then printf 'exists %s\n' "$path"; else printf 'missing %s\n' "$path"; fi
done
find /data/chenyiteng/models/fastwam/diffsynth -maxdepth 3 -type d -print 2>/dev/null | head -40
printf '%s\n' 'gpu_processes:'
nvidia-smi -i 2,3 --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray status | sed -n '1,16p'
