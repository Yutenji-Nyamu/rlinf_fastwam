#!/usr/bin/env bash
set +e

config_name="$1"
run_root="$2"
runtime_root="$3"
experiment_name="$4"
ray_address="$5"
driver_id="$6"

repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json

mkdir -p "$run_root" "$runtime_root" "/tmp/raydriver_${driver_id}" "/tmp/runtmp_${driver_id}"
cd "$repo"
export PYTHONPATH="$repo:$assets" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export EMBODIED_PATH="$repo/examples/embodiment" REPO_PATH="$repo" RLINF_CODE_WORKING_DIR="$repo"
export ROBOTWIN_PATH="$assets" ROBOTWIN_ASSETS_PATH="$assets" ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1 TORCHINDUCTOR_COMPILE_THREADS=1
export RAY_TMPDIR="/tmp/raydriver_${driver_id}" TMPDIR="/tmp/runtmp_${driver_id}" RAY_ADDRESS="$ray_address"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export RLT_LOG_ROOT="$run_root" ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export RLT_STAGE1_MODEL_PATH="$stage1_model" RLT_STAGE1_MANIFEST_PATH="$stage1_manifest"
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY

overrides=(
  "runner.logger.log_path=${run_root}"
  "runner.logger.experiment_name=${experiment_name}"
  "runner.max_steps=480"
  "runner.val_check_interval=25"
  "runner.save_interval=25"
  "runner.resume_dir=null"
  "env.train.task_config.save_path=${run_root}/robotwin_data/train"
  "env.eval.task_config.save_path=${run_root}/robotwin_data/eval"
)

git rev-parse HEAD >"$runtime_root/source_head.txt"
printf '%s\n' "$config_name" >"$runtime_root/config_name.txt"
printf '%s\n' "$ray_address" >"$runtime_root/ray_address.txt"
printf '%s\n' "$driver_id" >"$runtime_root/driver_id.txt"
printf '%q ' "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py --config-path "$repo/examples/embodiment/config" --config-name "$config_name" "${overrides[@]}" >"$runtime_root/exact_command.txt"
printf '\n' >>"$runtime_root/exact_command.txt"

"$venv/bin/python" -B examples/embodiment/train_embodied_agent.py --config-path "$repo/examples/embodiment/config" --config-name "$config_name" "${overrides[@]}" --cfg job --resolve >"$runtime_root/resolved.yaml" 2>"$runtime_root/compose.stderr.log"
rc=$?
printf '%s\n' "$rc" >"$runtime_root/compose_exit_code.txt"
test "$rc" -eq 0 || exit "$rc"
sha256sum "$runtime_root/resolved.yaml" >"$runtime_root/resolved.sha256"

date -Is >"$runtime_root/started_at.txt"
"$venv/bin/python" -B examples/embodiment/train_embodied_agent.py --config-path "$repo/examples/embodiment/config" --config-name "$config_name" "${overrides[@]}"
rc=$?
printf '%s\n' "$rc" >"$runtime_root/exit_code.txt"
date -Is >"$runtime_root/finished_at.txt"
exit "$rc"
