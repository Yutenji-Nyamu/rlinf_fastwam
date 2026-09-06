#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
out=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_dual_prelaunch_20260828_v1
cfg=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s2p0_mb256_warm20k_replay80k_gpu1_fresh480
cd "$repo"
test "$(git rev-parse HEAD)" = ff432cbf35e2ea9c3aa805b20fb180a846fc70e5
test "$(git status --short | wc -l)" -eq 1
git diff --check
export PYTHONPATH="$repo:$assets" EMBODIED_PATH="$repo/examples/embodiment" REPO_PATH="$repo" RLINF_CODE_WORKING_DIR="$repo"
export ROBOTWIN_PATH="$assets" ROBOTWIN_ASSETS_PATH="$assets" ROBOT_PLATFORM=ALOHA
export RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_dvac_pure_dual_compose_dummy
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
export RLT_STAGE1_MODEL_PATH=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
export RLT_STAGE1_MANIFEST_PATH=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
"$venv/bin/python" -B examples/embodiment/train_embodied_agent.py --config-path "$repo/examples/embodiment/config" --config-name "$cfg" --cfg job --resolve >"$out/${cfg}.yaml"
grep -A2 'component_placement:' "$out/${cfg}.yaml" | head -3
grep -F 'strength: 2.0' "$out/${cfg}.yaml"
git add examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s2p0_mb256_warm20k_replay80k_gpu1_fresh480.yaml
git commit -m 'config(rlt): place strong pure DVAC on gpu1'
git rev-parse HEAD
