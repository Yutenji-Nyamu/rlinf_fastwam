#!/usr/bin/env bash
set -euo pipefail

export_root=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1
experiment_root=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
repo_root=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
stage_dir="${export_root}/download_bundle_v1"
bundle_path="${export_root}/rlt_stage1_formal_high_info_20260729_v1.tar.gz"

test -d "${export_root}"
test -d "${experiment_root}"
test -d "${repo_root}"
test ! -e "${stage_dir}"
test ! -e "${bundle_path}"

mkdir -p \
  "${stage_dir}/formal_run/runtime" \
  "${stage_dir}/formal_run/tensorboard" \
  "${stage_dir}/artifact_acceptance_v1" \
  "${stage_dir}/artifact_acceptance_v2" \
  "${stage_dir}/source"

for name in \
  formal_resolved.yaml \
  source_config.yaml \
  dataset_manifest.json \
  run_provenance.tsv \
  early_health.json
do
  cp -a "${export_root}/${name}" "${stage_dir}/formal_run/${name}"
done

for name in \
  driver.log \
  resources.csv \
  exit_code.txt \
  started_at.txt \
  finished_at.txt
do
  cp -a "${export_root}/runtime/${name}" "${stage_dir}/formal_run/runtime/${name}"
done

tensorboard_file="$(
  find "${experiment_root}/tensorboard" -maxdepth 1 -type f \
    -name 'events.out.tfevents.*' -print -quit
)"
test -n "${tensorboard_file}"
cp -a "${tensorboard_file}" "${stage_dir}/formal_run/tensorboard/"

cp -a "${export_root}/artifact_acceptance_v1/." \
  "${stage_dir}/artifact_acceptance_v1/"
cp -a "${export_root}/artifact_acceptance_v2/." \
  "${stage_dir}/artifact_acceptance_v2/"

cp -a \
  "${repo_root}/toolkits/rlt/validate_robotwin_rlt_stage1_artifact.py" \
  "${stage_dir}/source/"
cp -a \
  "${repo_root}/local_scripts/remote_rlt_20260729_stage1_artifact_acceptance.sh" \
  "${stage_dir}/source/"
cp -a \
  "${repo_root}/local_scripts/remote_rlt_20260729_start_stage1_artifact_acceptance.sh" \
  "${stage_dir}/source/"

(
  cd "${repo_root}"
  git status --short --branch
  git rev-parse HEAD
) > "${stage_dir}/source/repo_state_at_packaging.txt"

(
  cd "${stage_dir}"
  find . -type f ! -name CONTENTS_SHA256.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "${stage_dir}/CONTENTS_SHA256.txt"

(
  cd "${export_root}"
  tar -czf "${bundle_path}" "$(basename "${stage_dir}")"
)

sha256sum "${bundle_path}"
du -h "${bundle_path}"
find "${stage_dir}" -type f | sort
