#!/usr/bin/env bash
set -u
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4
CKPT="$RUN/robotwin_grpo_openpi_dvac_global_z_matched/checkpoints/global_step_30"
date --iso-8601=seconds
if [[ -d "$CKPT" ]]; then
  echo checkpoint30=yes
  printf 'files='; find "$CKPT" -type f | wc -l
  printf 'dvac_sidecars='; find "$CKPT" -type f -name 'dvac_state_rank*.json' | wc -l
  du -sh "$CKPT"
else
  echo checkpoint30=no
fi
free -h | sed -n '1,3p'
