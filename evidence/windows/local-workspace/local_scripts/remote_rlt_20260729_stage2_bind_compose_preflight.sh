#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
evidence_root=/root/autodl-tmp/experiment_exports/rlt_stage2_pre_smoke_20260729_v1
smoke_root=/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
pilot_pending_root=/root/autodl-tmp/experiments/rlt_stage2_pilot_PENDING_APPROVAL
fresh_name=robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1
resume_name=robotwin_adjust_bottle_rlt_stage2_smoke_resume_v1
fresh_checkpoint="${smoke_root}/${fresh_name}/checkpoints/global_step_1"

stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
stage1_manifest_id=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
stage1_manifest_sha256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
norm_stats_sha256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

test "$(git -C "${repo}" branch --show-current)" = codex/rlt-pi0-robotwin
test "$(git -C "${repo}" rev-parse HEAD)" = 6df42bf488ef10d9c7eb2f89584bc5ab7543a08a
test ! -e "${evidence_root}"
test ! -e "${smoke_root}"
test ! -e "${pilot_pending_root}"

if pgrep -af \
  'train_embodied_agent|ray::|raylet|gcs_server|robotwin_adjust_bottle_rlt_stage2' \
  | grep -v -E 'pgrep -af|stage2_bind_compose_preflight' >/dev/null
then
  printf '%s\n' "Refusing preflight while an RLT/Ray process is active." >&2
  exit 1
fi

mkdir -p "${evidence_root}/source_configs"
date --iso-8601=seconds >"${evidence_root}/started_at.txt"
start_epoch="$(date +%s)"

{
  date --iso-8601=seconds
  git -C "${repo}" status --short --branch
  git -C "${repo}" rev-parse HEAD
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  free -b
  df -B1 /root/autodl-tmp
} >"${evidence_root}/resources_and_source_before.txt"

export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export REPO_PATH="${repo}"
export ROBOTWIN_ASSETS_PATH="${assets}"
export ROBOTWIN_PI0_NORM_STATS_PATH="${norm_stats}"
export RLT_STAGE1_MODEL_PATH="${stage1_model}"
export RLT_STAGE1_MANIFEST_PATH="${stage1_manifest}"
export RLT_STAGE1_MANIFEST_ID="${stage1_manifest_id}"
export RLT_STAGE1_MANIFEST_SHA256="${stage1_manifest_sha256}"
export RLT_NORM_STATS_SHA256="${norm_stats_sha256}"

"${venv}/bin/python" -B \
  "${repo}/toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py" \
  --manifest-path "${stage1_manifest}" \
  --manifest-id "${stage1_manifest_id}" \
  --manifest-sha256 "${stage1_manifest_sha256}" \
  --stage1-model-path "${stage1_model}" \
  --norm-stats-path "${norm_stats}" \
  --norm-stats-sha256 "${norm_stats_sha256}" \
  --canonical-adapter-version robotwin_aloha_canonical_v1 \
  --action-horizon 50 \
  --action-chunk 10 \
  --action-dim 14 \
  --z-rl-dim 2048 \
  --prefix-seq-len 768 \
  --prefix-dim 2048 \
  --output "${evidence_root}/stage1_binding_preflight.json" \
  >"${evidence_root}/stage1_binding_preflight.stdout"

compose_embodied() {
  config_name="$1"
  output_path="$2"
  shift 2
  export EMBODIED_PATH="${repo}/examples/embodiment"
  "${venv}/bin/python" -B \
    "${repo}/examples/embodiment/train_embodied_agent.py" \
    --config-path "${repo}/examples/embodiment/config" \
    --config-name "${config_name}" \
    "$@" \
    --cfg job \
    --resolve >"${output_path}"
}

export RLT_LOG_ROOT="${pilot_pending_root}"
compose_embodied \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp \
  "${evidence_root}/formal_bound_resolved.yaml"

export RLT_LOG_ROOT="${smoke_root}"
compose_embodied \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke \
  "${evidence_root}/fresh_bound_resolved.yaml" \
  "runner.logger.experiment_name=${fresh_name}"
compose_embodied \
  robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke \
  "${evidence_root}/resume_bound_resolved.yaml" \
  "runner.max_steps=2" \
  "runner.resume_dir=${fresh_checkpoint}" \
  "runner.logger.experiment_name=${resume_name}"

"${venv}/bin/python" -B \
  "${repo}/toolkits/rlt/audit_robotwin_rlt_stage2_resolved.py" \
  --formal "${evidence_root}/formal_bound_resolved.yaml" \
  --fresh "${evidence_root}/fresh_bound_resolved.yaml" \
  --resume "${evidence_root}/resume_bound_resolved.yaml" \
  --stage1-model-path "${stage1_model}" \
  --manifest-path "${stage1_manifest}" \
  --manifest-id "${stage1_manifest_id}" \
  --manifest-sha256 "${stage1_manifest_sha256}" \
  --norm-stats-path "${norm_stats}" \
  --norm-stats-sha256 "${norm_stats_sha256}" \
  --resume-dir "${fresh_checkpoint}" \
  --resume-experiment-name "${resume_name}" \
  --output "${evidence_root}/resolved_contract_audit.json" \
  >"${evidence_root}/resolved_contract_audit.stdout"

cp -a \
  "${repo}/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml" \
  "${evidence_root}/source_configs/"
cp -a \
  "${repo}/examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml" \
  "${evidence_root}/source_configs/"

{
  printf 'repo\t%s\n' "${repo}"
  printf 'head\t%s\n' "$(git -C "${repo}" rev-parse HEAD)"
  printf 'stage1_model\t%s\n' "${stage1_model}"
  printf 'stage1_manifest\t%s\n' "${stage1_manifest}"
  printf 'stage1_manifest_id\t%s\n' "${stage1_manifest_id}"
  printf 'stage1_manifest_sha256\t%s\n' "${stage1_manifest_sha256}"
  printf 'norm_stats\t%s\n' "${norm_stats}"
  printf 'norm_stats_sha256\t%s\n' "${norm_stats_sha256}"
  printf 'smoke_root\t%s\n' "${smoke_root}"
  printf 'fresh_name\t%s\n' "${fresh_name}"
  printf 'resume_name\t%s\n' "${resume_name}"
  printf 'fresh_checkpoint\t%s\n' "${fresh_checkpoint}"
} >"${evidence_root}/bindings.tsv"

git -C "${repo}" diff -- \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_robotwin_rlt_contract.py \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml \
  >"${evidence_root}/stage2_hardening.diff"

{
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  free -b
  df -B1 /root/autodl-tmp
} >"${evidence_root}/resources_after.txt"

finish_epoch="$(date +%s)"
date --iso-8601=seconds >"${evidence_root}/finished_at.txt"
printf '%s\n' "$((finish_epoch - start_epoch))" >"${evidence_root}/duration_seconds.txt"

(
  cd "${evidence_root}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum
) >"${evidence_root}/SHA256SUMS"

printf '%s\n' STAGE2_BIND_COMPOSE_PREFLIGHT_OK
cat "${evidence_root}/duration_seconds.txt"
sha256sum \
  "${evidence_root}/stage1_binding_preflight.json" \
  "${evidence_root}/formal_bound_resolved.yaml" \
  "${evidence_root}/fresh_bound_resolved.yaml" \
  "${evidence_root}/resume_bound_resolved.yaml" \
  "${evidence_root}/resolved_contract_audit.json"
