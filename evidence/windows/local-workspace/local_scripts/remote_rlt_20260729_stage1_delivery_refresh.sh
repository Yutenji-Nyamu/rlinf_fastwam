set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
DSRL_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf
CHECKPOINT=/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/s1a/robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1/checkpoints/global_step_2
CONTRACT=/root/autodl-tmp/experiment_exports/rlt_stage1_smoke_20260729_v1/lr_scheduler_contract.json

date -Is
printf '%s\n' 'RLT_GIT'
git -C "$RLT_ROOT" branch --show-current
git -C "$RLT_ROOT" rev-parse HEAD
git -C "$RLT_ROOT" status --short
git -C "$RLT_ROOT" rev-list --left-right --count HEAD...@{upstream}
git -C "$RLT_ROOT" ls-remote personal refs/heads/codex/rlt-pi0-robotwin
printf '%s\n' 'DSRL_GIT'
git -C "$DSRL_ROOT" branch --show-current
git -C "$DSRL_ROOT" status --short
printf '%s\n' 'GPU'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
printf '%s\n' 'RAM'
free -h
printf '%s\n' 'DISK'
df -hT /root/autodl-tmp
printf '%s\n' 'PROCESSES'
pgrep -af 'train_vla_sft|ray::|raylet|gcs_server|robotwin' || true
printf '%s\n' 'ARTIFACTS'
test -f "$CHECKPOINT/actor/dcp_checkpoint/.metadata"
test -f "$CHECKPOINT/actor/model_state_dict/full_weights.pt"
test -f "$CONTRACT"
du -sh "$CHECKPOINT"
sha256sum \
  "$RLT_ROOT/examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml" \
  "$RLT_ROOT/docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/lr_scheduler_contract.json" \
  "$CONTRACT"
