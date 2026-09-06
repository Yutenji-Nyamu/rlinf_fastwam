#!/usr/bin/env bash
set +e

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
run_root=/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
runtime_root=/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
experiment_name=robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_formal_fresh480_v1
config_name=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2_fresh480
stage1_model=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
stage1_manifest=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
norm_stats=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json

cd "$repo"
export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="${repo}/examples/embodiment"
export REPO_PATH="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=0,1
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

git rev-parse HEAD >"${runtime_root}/source_head.txt"
date -Is >"${runtime_root}/started_at.txt"
"${venv}/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "${repo}/examples/embodiment/config" \
  --config-name "$config_name" \
  "runner.logger.log_path=${run_root}" \
  "runner.logger.experiment_name=${experiment_name}"
rc=$?
printf '%s\n' "$rc" >"${runtime_root}/exit_code.txt"
date -Is >"${runtime_root}/finished_at.txt"
{
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
  printf 'cgroup_memory_current='; cat /sys/fs/cgroup/memory.current
  cat /sys/fs/cgroup/memory.events
} >"${runtime_root}/resources_after.txt"
exit "$rc"
