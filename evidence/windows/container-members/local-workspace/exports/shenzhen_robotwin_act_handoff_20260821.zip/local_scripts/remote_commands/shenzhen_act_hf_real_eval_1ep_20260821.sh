#!/usr/bin/env bash
set -euo pipefail

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin

ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
HF_LEAF=/data/chenyiteng/models/robotwin2-hf-a967b852/act_ckpt/act-adjust_bottle/demo_clean-50
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
EVAL_CFG="$RUN/configs/sz_adjust_bottle_act_1ep.yml"
OUT="$RUN/08_act_hf_eval_1ep_scheduler"

test ! -e "$OUT"
test -s "$HF_LEAF/policy_last.ckpt"
test -s "$HF_LEAF/dataset_stats.pkl"
printf '%s  %s\n' 783a69199e287419c084ee004bd077665823dbd6ffa3a7b30efd630d0f6e6bc9 "$EVAL_CFG" | sha256sum --check -

cd "$ROBOTWIN"
COMMON=(
  --config "$EVAL_CFG"
  --policy-name ACT
  --ckpt-name "$HF_LEAF"
  --env-cfg-type aloha_agilex
  --policy-conda-env act
  --eval-env-conda-env RoboTwin
  --bench-name RoboTwin
  --action-type joint
  --seed 0
  --task-config demo_clean
  --test-num 1
  --num-workers 1
  --expert-check
  --output-dir "$OUT"
  --stream-output
)

bash scripts/eval_policy.sh multitask "${COMMON[@]}" --dry-run 2>&1 | tee "$RUN/08_act_hf_eval_1ep_dry_run.log"
timeout --signal=INT --kill-after=120s 2700s \
  bash scripts/eval_policy.sh multitask "${COMMON[@]}" 2>&1 | \
  tee "$RUN/08_act_hf_eval_1ep_scheduler.log"
