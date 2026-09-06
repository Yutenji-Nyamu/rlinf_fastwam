#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
VENV=/root/autodl-tmp/RLinf/.venv
EXPORT_ROOT=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
OUTPUT_ROOT="${EXPORT_ROOT}/artifact_acceptance_v2"
ENDPOINT=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
BASE_MODEL=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
NORM_STATS="${BASE_MODEL}/physical-intelligence/robotwin/norm_stats.json"
DATASET=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1

if [[ -e "${OUTPUT_ROOT}/started_at.txt" ]]; then
  echo "refusing to overwrite an existing acceptance run: ${OUTPUT_ROOT}" >&2
  exit 90
fi
mkdir -p "${OUTPUT_ROOT}"
date --iso-8601=seconds >"${OUTPUT_ROOT}/started_at.txt"

{
  echo "worktree=${WORKTREE}"
  echo "branch=$(git -C "${WORKTREE}" branch --show-current)"
  echo "head=$(git -C "${WORKTREE}" rev-parse HEAD)"
  echo "tool_sha256=$(sha256sum "${WORKTREE}/toolkits/rlt/validate_robotwin_rlt_stage1_artifact.py" | awk '{print $1}')"
  echo "endpoint=${ENDPOINT}"
  echo "base_model=${BASE_MODEL}"
  echo "norm_stats=${NORM_STATS}"
  echo "dataset=${DATASET}"
} >"${OUTPUT_ROOT}/launch_provenance.txt"

{
  echo "phase=before"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  free -b
} >"${OUTPUT_ROOT}/resources_before_after.txt"

set +e
(
  cd "${WORKTREE}"
  export PYTHONPATH="${WORKTREE}:/root/autodl-tmp/RoboTwin_RLinf"
  export PYTHONDONTWRITEBYTECODE=1
  export CUDA_VISIBLE_DEVICES=0
  export JAX_PLATFORMS=cpu
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=1
  timeout --signal=TERM --kill-after=60s 3600s \
    "${VENV}/bin/torchrun" --standalone --nproc-per-node=1 \
    toolkits/rlt/validate_robotwin_rlt_stage1_artifact.py \
    --endpoint "${ENDPOINT}" \
    --base-model "${BASE_MODEL}" \
    --norm-stats "${NORM_STATS}" \
    --dataset "${DATASET}" \
    --dataset-manifest "${EXPORT_ROOT}/dataset_manifest.json" \
    --formal-resolved-config "${EXPORT_ROOT}/formal_resolved.yaml" \
    --source-config "${EXPORT_ROOT}/source_config.yaml" \
    --training-provenance "${EXPORT_ROOT}/run_provenance.tsv" \
    --output-json "${OUTPUT_ROOT}/validation.json" \
    --artifact-manifest "${OUTPUT_ROOT}/stage1_artifact_manifest.json" \
    --manifest-id robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1 \
    --device cuda:0 \
    --batch-size 4 \
    --seed 0
) >"${OUTPUT_ROOT}/driver.log" 2>"${OUTPUT_ROOT}/stderr.log"
exit_code=$?
set -e

printf '%s\n' "${exit_code}" >"${OUTPUT_ROOT}/exit_code.txt"
date --iso-8601=seconds >"${OUTPUT_ROOT}/finished_at.txt"
{
  echo "phase=after"
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  free -b
} >>"${OUTPUT_ROOT}/resources_before_after.txt"

if [[ -f "${OUTPUT_ROOT}/validation.json" ]]; then
  sha256sum \
    "${OUTPUT_ROOT}/validation.json" \
    "${OUTPUT_ROOT}/stage1_artifact_manifest.json" \
    >"${OUTPUT_ROOT}/artifact_sha256.txt"
fi

exit "${exit_code}"
