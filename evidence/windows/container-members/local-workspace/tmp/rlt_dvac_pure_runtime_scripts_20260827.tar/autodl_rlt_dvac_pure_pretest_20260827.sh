#!/usr/bin/env bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_pure
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
out=/root/autodl-tmp/experiment_exports/rlt_dvac_pure_pretest_20260827
control=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_mb256_warm20k_replay80k_control
pure=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_dvac_pure_reference_bc_s0p5_mb256_warm20k_replay80k_fresh480

mkdir -p "$out"
cd "$repo"
export PYTHONPATH="$repo:$assets"
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export ROBOTWIN_PATH="$assets"
export ROBOTWIN_ASSETS_PATH="$assets"
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
export RLT_STAGE1_MODEL_PATH=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
export RLT_STAGE1_MANIFEST_PATH=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

"$venv/bin/ruff" format \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py
"$venv/bin/ruff" check \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py
"$venv/bin/python" -m py_compile \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$venv/bin/python" -m pytest -q tests/unit_tests/test_rlt_dvac_weighting.py

RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_pure_compose \
  "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name "$control" --cfg job --resolve >"$out/control.resolved.yaml"
RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_pure_compose \
  "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name "$pure" --cfg job --resolve >"$out/pure.resolved.yaml"

"$venv/bin/python" - "$out/control.resolved.yaml" "$out/pure.resolved.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    control = yaml.safe_load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    pure = yaml.safe_load(handle)

def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, path))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}

cflat = flatten(control)
pflat = flatten(pure)
all_keys = sorted(set(cflat) | set(pflat))
diffs = [key for key in all_keys if cflat.get(key) != pflat.get(key)]
allowed = (
    "algorithm.rlt_dvac.",
    "rollout.rlt_feature_model.openpi.rlt_dvac_mode",
    "runner.logger.experiment_name",
)
unexpected = [key for key in diffs if not key.startswith(allowed)]
assert not unexpected, unexpected

cfg = pure["algorithm"]["rlt_dvac"]
assert cfg["mode"] == "apply"
assert cfg["application"] == "success_episode_bc"
assert cfg["success_target"] == "reference"
assert cfg["selected_l"] == 3
assert cfg["applied_horizon"] == 10
assert float(cfg["z_clip"]) == 2.0
assert float(cfg["strength"]) == 0.5
assert float(cfg["success_scale"]) == 1.0
assert pure["cluster"]["component_placement"] == {"actor, env, rollout": 0}
assert pure["env"]["train"]["total_num_envs"] == 8
assert pure["env"]["eval"]["total_num_envs"] == 4
assert pure["env"]["eval"]["rollout_epoch"] == 5
assert pure["actor"]["global_batch_size"] == 512
assert pure["actor"]["micro_batch_size"] == 256
assert pure["algorithm"]["rlt_schedule"]["warmup_min_size"] == 20000
assert pure["algorithm"]["replay_buffer"]["cache_size"] == 80000
assert pure["runner"]["max_steps"] == 480
print("RLT_DVAC_PURE_CONFIG_CONTRACT_OK")
print("resolved_diff_keys=")
for key in diffs:
    print(key)
PY

git diff --check
git status --short
git diff --stat
echo RLT_DVAC_PURE_PRETEST_OK
