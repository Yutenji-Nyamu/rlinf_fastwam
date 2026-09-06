#!/usr/bin/env bash
set -u
SRC=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
WAM=/root/autodl-tmp/idea2_dvac_train_wamppo
SFT=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
V2=/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
date --iso-8601=seconds
printf 'RLINF_HEAD\t'; git -C "$SRC" rev-parse HEAD
printf 'RLINF_STATUS\t'; git -C "$SRC" status --porcelain | wc -l
printf 'WAM_HEAD\t'; git -C "$WAM" rev-parse HEAD
printf 'WAM_STATUS\t'; git -C "$WAM" status --porcelain | wc -l
sha256sum \
 "$SRC/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml" \
 "$SRC/rlinf/envs/robotwin/seeds/eval_seeds.json" \
 "$SFT/physical-intelligence/robotwin/norm_stats.json"
printf '%s\n' 'V2_CHECKPOINTS'
find "$V2/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort -V
