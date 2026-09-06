#!/usr/bin/env bash
set -u
RUN=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823
RUNTIME=/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823

echo TOP_LEVEL_RUN
find "$RUN" -maxdepth 2 -mindepth 1 -printf '%y\t%s\t%p\n' | sort -k3
echo TOP_LEVEL_RUNTIME
find "$RUNTIME" -maxdepth 2 -mindepth 1 -printf '%y\t%s\t%p\n' | sort -k3
echo AUXILIARY_FILES
find "$RUN" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.yaml' -o -name '*.mp4' -o -name 'events.out.tfevents*' \) -printf '%s\t%p\n' | sort -k2
echo CHECKPOINT_40_CONTENTS
find "$RUN" -path '*/checkpoints/global_step_40/*' -type f -printf '%s\t%p\n' | sort -nr | head -30
echo LATEST_NPZ_PER_RANK
for rank in 00 01; do
  dir="$RUN/dvac_train/actor_rank${rank}"
  printf 'rank%s files=' "$rank"
  find "$dir" -maxdepth 1 -name 'rollout_step*.npz' -type f | wc -l
  find "$dir" -maxdepth 1 -name 'rollout_step*.npz' -type f -printf '%f %s\n' | sort | tail -3
done
