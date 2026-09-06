#!/usr/bin/env bash
set -u
SRC=/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
printf '%s\n' '=== CONVERTER ==='
sed -n '1,260p' "$SRC/rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py"
printf '%s\n' '=== OPENPI MODEL LOAD ==='
sed -n '1,130p' "$SRC/rlinf/models/embodiment/openpi/__init__.py"
printf '%s\n' '=== SAVE/LOAD CORE ==='
sed -n '180,355p' "$SRC/rlinf/hybrid_engines/fsdp/strategy/base.py"
printf '%s\n' '=== DCP TYPE METADATA QUICK ==='
for p in \
 /root/autodl-tmp/RLinf/logs/20260715_132507-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline-env16-rollout16-g8-step0-to-100/robotwin_grpo_openpi_2gpu_env16_rollout16_g8_baseline/checkpoints/global_step_100/actor/dcp_checkpoint/.metadata \
 /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821/idea2_dvac_apply_formal_100step_2gpu16env_20260821/checkpoints/global_step_50/actor/dcp_checkpoint/.metadata \
 /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822/checkpoints/global_step_20/actor/dcp_checkpoint/.metadata; do
  printf '%s\t' "$p"
  stat -c '%s bytes' "$p" 2>/dev/null || echo missing
done
