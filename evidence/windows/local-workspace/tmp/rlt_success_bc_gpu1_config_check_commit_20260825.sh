set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
cfg=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_gpu1_fresh480
out=/root/autodl-tmp/experiment_exports/rlt_success_bc_gpu1_config_check_20260825
cd "$repo"
mkdir -p "$out"
export PYTHONPATH="$repo:$assets"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
export RLT_STAGE1_MODEL_PATH=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
export RLT_STAGE1_MANIFEST_PATH=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_success_bc_gpu1_check \
  "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" --config-name "$cfg" \
  --cfg job --resolve >"$out/resolved.yaml"
"$venv/bin/python" - "$out/resolved.yaml" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
assert cfg["cluster"]["component_placement"] == {"actor, env, rollout": 1}
assert cfg["actor"]["global_batch_size"] == 512
assert cfg["actor"]["micro_batch_size"] == 128
assert cfg["algorithm"]["rlt_dvac"]["application"] == "success_episode_bc"
print("GPU1_OVERLAY_OK")
PY
git add examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_gpu1_fresh480.yaml
git diff --cached --check
git commit -m 'config(rlt): add GPU1 success BC runtime overlay'
git push personal codex/rlt-dvac-success-episode-bc
git rev-parse HEAD
git status --short --branch
