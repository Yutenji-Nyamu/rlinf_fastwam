set -euo pipefail
repo=/root/autodl-tmp/RLinf_rlt_dvac_success_bc
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
cfg=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_success_episode_bc_dvac_w0to2_fresh480
control=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_control
out=/root/autodl-tmp/experiment_exports/rlt_success_bc_pretest_20260825

cd "$repo"
mkdir -p "$out"
echo '[STATUS]'
git status --short
git diff --check
git diff --stat

if test -x "$venv/bin/ruff"; then
  "$venv/bin/ruff" format --check \
    rlinf/algorithms/rlt/dvac_weighting.py \
    rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
    tests/unit_tests/test_rlt_dvac_weighting.py
  "$venv/bin/ruff" check \
    rlinf/algorithms/rlt/dvac_weighting.py \
    rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
    tests/unit_tests/test_rlt_dvac_weighting.py
fi

"$venv/bin/python" -m py_compile \
  rlinf/algorithms/rlt/dvac_weighting.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  tests/unit_tests/test_rlt_dvac_weighting.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$repo:$assets" \
  "$venv/bin/python" -m pytest -q tests/unit_tests/test_rlt_dvac_weighting.py

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
export HYDRA_FULL_ERROR=1
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json
export RLT_STAGE1_MODEL_PATH=/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
export RLT_STAGE1_MANIFEST_PATH=/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2/stage1_artifact_manifest.json
export RLT_STAGE1_MANIFEST_ID=robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
export RLT_STAGE1_MANIFEST_SHA256=6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
export RLT_NORM_STATS_SHA256=649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a

RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_success_bc_pretest \
  "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name "$cfg" --cfg job --resolve >"$out/method.resolved.yaml"
RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_control_pretest \
  "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$repo/examples/embodiment/config" \
  --config-name "$control" --cfg job --resolve >"$out/control.resolved.yaml"

"$venv/bin/python" - "$out/control.resolved.yaml" "$out/method.resolved.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    control = yaml.safe_load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    method = yaml.safe_load(handle)

def get(data, path):
    for key in path.split("."):
        data = data[key]
    return data

same = (
    "cluster.component_placement",
    "runner.max_steps",
    "runner.resume_dir",
    "runner.val_check_interval",
    "runner.save_interval",
    "env.train.total_num_envs",
    "env.train.rollout_epoch",
    "env.train.auto_reset",
    "env.eval.total_num_envs",
    "env.eval.rollout_epoch",
    "actor.global_batch_size",
    "actor.micro_batch_size",
    "actor.seed",
    "actor.optim",
    "actor.critic_optim",
    "algorithm.update_epoch",
    "algorithm.critic_actor_ratio",
    "algorithm.rlt_schedule",
    "algorithm.replay_buffer",
    "algorithm.actor_weight_schedule",
    "rollout.rlt_feature_model.model_path",
)
for path in same:
    assert get(control, path) == get(method, path), path

dvac = get(method, "algorithm.rlt_dvac")
assert "rlt_dvac" not in control["algorithm"]
assert dvac["mode"] == "apply"
assert dvac["application"] == "success_episode_bc"
assert float(dvac["z_clip"]) == 2.0
assert float(dvac["strength"]) == 0.25
assert float(dvac["success_scale"]) == 1.0
assert int(dvac["selected_l"]) == 3
assert int(dvac["applied_horizon"]) == 10
assert get(control, "cluster.component_placement") == {"actor, env, rollout": 0}
assert int(get(control, "actor.global_batch_size")) == 512
assert int(get(control, "actor.micro_batch_size")) == 128
print("SUCCESS_BC_SINGLE_GPU_AB_CONTRACT_OK")
print("derived_gradient_accumulation_world1=4")
PY

sha256sum "$out/control.resolved.yaml" "$out/method.resolved.yaml"
