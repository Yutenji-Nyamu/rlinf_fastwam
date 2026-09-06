#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
DSRL_ROOT=/root/autodl-tmp/RLinf_fastwam_rlinf
SOURCE_ROOT=/root/autodl-tmp/datasets/robotwin2/source/9dc9299c163db059931898a9f0852098a61155a1/dataset/adjust_bottle
CANONICAL_TARGET=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
FORMAL_ROOT=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
EXPORT_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
MOTUS_OLD=/root/autodl-tmp/RoboTwin/policy/Motus_old_20260618_111133

printf 'OBSERVED_AT\n'
date --iso-8601=seconds
printf 'IDENTITY\n'
id
hostname

printf 'RLT_GIT\n'
git -C "$RLT_ROOT" branch --show-current
git -C "$RLT_ROOT" rev-parse HEAD
git -C "$RLT_ROOT" status --short
git -C "$RLT_ROOT" rev-list --left-right --count HEAD...@{upstream}
git -C "$RLT_ROOT" ls-remote personal refs/heads/codex/rlt-pi0-robotwin

printf 'DSRL_GIT\n'
git -C "$DSRL_ROOT" branch --show-current
git -C "$DSRL_ROOT" rev-parse HEAD
git -C "$DSRL_ROOT" status --short

printf 'RESOURCES\n'
df -hT /root/autodl-tmp
free -h
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

printf 'RELEVANT_PROCESSES\n'
pgrep -af 'train_vla_sft|ray::|raylet|gcs_server|robotwin|rlt_stage1' || true

printf 'FORMAL_TARGET_GUARDS\n'
for path in "$CANONICAL_TARGET" "$FORMAL_ROOT" "$EXPORT_ROOT"; do
  if [ -e "$path" ] || [ -L "$path" ]; then
    printf 'EXISTS\t%s\n' "$path"
    du -sx --block-size=1 -- "$path" || true
  else
    printf 'ABSENT\t%s\n' "$path"
  fi
done

printf 'SOURCE_FILES\n'
find "$SOURCE_ROOT" -maxdepth 2 -type f \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -k3
printf 'SOURCE_HASHES\n'
find "$SOURCE_ROOT" -maxdepth 2 -type f -name '*.zip' -print0 | \
  sort -z | xargs -0 -r sha256sum

printf 'RLINF_RUN_CHECKPOINTS\n'
shopt -s nullglob
rlinf_runs=(
  /root/autodl-tmp/RLinf/logs/*20260714_170304*
  /root/autodl-tmp/RLinf/logs/*20260714_181545*
  /root/autodl-tmp/RLinf/logs/*20260715_113256*
  /root/autodl-tmp/RLinf/logs/*20260715_132507*
)
test "${#rlinf_runs[@]}" -eq 4
for run in "${rlinf_runs[@]}"; do
  printf 'RUN\t%s\n' "$run"
  find "$run" -xdev -type d -name 'global_step_*' -print0 | \
    sort -zV | while IFS= read -r -d '' checkpoint; do
      du -sx --block-size=1 -- "$checkpoint"
    done
done

printf 'MOTUS_EXPERIMENT_CHECKPOINTS\n'
test -d "$MOTUS_OLD/logs_single_20260602_170538"
test -d "$MOTUS_OLD/logs_single_20260601_082941"
for run in \
  "$MOTUS_OLD/logs_single_20260602_170538" \
  "$MOTUS_OLD/logs_single_20260601_082941"; do
  printf 'RUN\t%s\n' "$run"
  find "$run" -xdev -type f -name '*.pt' \
    -printf '%T@\t%s\t%p\n' | sort -n
done

printf 'MOTUS_OFFICIAL_WEIGHT_SHIELD\n'
find /root/autodl-tmp/RoboTwin/policy/Motus \
  -xdev -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) \
  -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' | sort -nr

printf 'CONFIG_HASHES\n'
sha256sum \
  "$RLT_ROOT/examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml" \
  "$RLT_ROOT/docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/lr_scheduler_contract.json" \
  /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
