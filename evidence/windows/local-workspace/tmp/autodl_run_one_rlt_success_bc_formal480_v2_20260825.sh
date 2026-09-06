#!/usr/bin/env bash
set +e

expected_gpu="$1"
config_name="$2"
run_root="$3"
runtime_root="$4"
experiment_name="$5"
ray_address="$6"
driver_id="$7"

repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
driver_ray_tmp="/tmp/raydriver_${driver_id}"
driver_tmp="/tmp/runtmp_${driver_id}"

mkdir -p "$run_root" "$runtime_root" "$driver_ray_tmp" "$driver_tmp"
cd "$repo"
export PYTHONPATH="$repo:$assets"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export RLINF_CODE_WORKING_DIR="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
export TORCHINDUCTOR_COMPILE_THREADS=1
export RAY_TMPDIR="$driver_ray_tmp"
export TMPDIR="$driver_tmp"
export RAY_ADDRESS="$ray_address"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RLT_LOG_ROOT="$run_root"
export ROBOTWIN_PI0_NORM_STATS_PATH="$norm_stats"
export RLT_STAGE1_MODEL_PATH="$stage1_model"
export RLT_STAGE1_MANIFEST_PATH="$stage1_manifest"
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
printf '%s\n' "$expected_gpu" >"$runtime_root/expected_physical_gpu.txt"
printf '%s\n' "$config_name" >"$runtime_root/config_name.txt"
printf '%s\n' "$ray_address" >"$runtime_root/ray_address.txt"
printf '%s\n' "$driver_id" >"$runtime_root/driver_id.txt"
{
  printf '%q ' "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
    --config-path "$repo/examples/embodiment/config" \
    --config-name "$config_name" "${overrides[@]}"
  printf '\n'
} >"$runtime_root/exact_command.txt"

"$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name "$config_name" "${overrides[@]}" --cfg job --resolve \
  >"$runtime_root/resolved.yaml" 2>"$runtime_root/compose.stderr.log"
compose_rc=$?
printf '%s\n' "$compose_rc" >"$runtime_root/compose_exit_code.txt"
if [ "$compose_rc" -ne 0 ]; then
  exit "$compose_rc"
fi
sha256sum "$runtime_root/resolved.yaml" >"$runtime_root/resolved.sha256"

date -Is >"$runtime_root/started_at.txt"
"$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name "$config_name" "${overrides[@]}"
rc=$?
printf '%s\n' "$rc" >"$runtime_root/exit_code.txt"
date -Is >"$runtime_root/finished_at.txt"
exit "$rc"
