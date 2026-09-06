set -eu
for ck in \
  /data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-action-adv-fix-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1/robotwin_dvac_action_adv_fix_w0p5to1p5_formal100_2gpu64x4_b1024_fixed32_eval5_phys45_v1/checkpoints/global_step_60 \
  /data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-st-global-z-w0p8to1p2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-localshard-v2/robotwin_dvac_st_global_z_w0p8to1p2_formal100_2gpu64x4_b1024_fixed32_eval5_phys67_localshard_v2/checkpoints/global_step_40 \
  /data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2/robotwin_grpo_control_2gpu64x4_b1024_fixed32_eval5_v2/checkpoints/global_step_90; do
  echo "CK|$ck"
  find "$ck" -xdev -type f -printf '%s|%p\n' | sort -t'|' -k2,2
done
