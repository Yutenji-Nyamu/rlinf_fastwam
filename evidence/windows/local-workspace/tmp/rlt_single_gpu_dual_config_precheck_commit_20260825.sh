#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_rlt_teacher_dvac
venv=/root/autodl-tmp/RLinf/.venv
assets=/root/autodl-tmp/RoboTwin_RLinf
cfg_dir="$repo/examples/embodiment/config"
control=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_control
dvac=robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_teacher_dvac_w05to15_fresh480
out=/root/autodl-tmp/experiment_exports/rlt_single_gpu_dual480_20260825_preflight

mkdir -p "$out"
cd "$repo"

echo '[BEFORE_STATUS]'
git status --short
git diff --check
git diff -- examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_control.yaml examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_teacher_dvac_w05to15_fresh480.yaml

export PYTHONPATH="${repo}:${assets}"
export PYTHONDONTWRITEBYTECODE=1
export EMBODIED_PATH="${repo}/examples/embodiment"
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

RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_single_gpu_control_fresh480_20260825_v1 \
  "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$cfg_dir" --config-name "$control" --cfg job --resolve \
  >"$out/control.resolved.yaml"

RLT_LOG_ROOT=/root/autodl-tmp/experiments/rlt_single_gpu_teacher_dvac_w05to15_fresh480_20260825_v1 \
  "$venv/bin/python" -B examples/embodiment/train_embodied_agent.py \
  --config-path "$cfg_dir" --config-name "$dvac" --cfg job --resolve \
  >"$out/dvac.resolved.yaml"

"$venv/bin/python" - "$out/control.resolved.yaml" "$out/dvac.resolved.yaml" "$out/contract_diff.json" <<'PY'
import json
import sys
import yaml

control_path, dvac_path, output_path = sys.argv[1:]
with open(control_path, encoding="utf-8") as f:
    control = yaml.safe_load(f)
with open(dvac_path, encoding="utf-8") as f:
    dvac = yaml.safe_load(f)

def select(data, path):
    for key in path.split('.'):
        data = data[key]
    return data

common = {
    "placement": select(control, "cluster.component_placement"),
    "max_steps": select(control, "runner.max_steps"),
    "train_envs": select(control, "env.train.total_num_envs"),
    "train_epochs": select(control, "env.train.rollout_epoch"),
    "eval_envs": select(control, "env.eval.total_num_envs"),
    "eval_epochs": select(control, "env.eval.rollout_epoch"),
    "global_batch": select(control, "actor.global_batch_size"),
    "micro_batch": select(control, "actor.micro_batch_size"),
    "warmup_min_size": select(control, "algorithm.rlt_schedule.warmup_min_size"),
    "replay_cache": select(control, "algorithm.replay_buffer.cache_size"),
    "replay_window": select(control, "algorithm.replay_buffer.sample_window_size"),
    "update_epoch": select(control, "algorithm.update_epoch"),
    "critic_actor_ratio": select(control, "algorithm.critic_actor_ratio"),
    "max_updates_per_cycle": select(control, "algorithm.rlt_schedule.max_updates_per_train_step"),
    "actor_seed": select(control, "actor.seed"),
    "teacher_h": select(control, "rollout.rlt_feature_model.openpi.action_horizon"),
    "teacher_m": select(control, "rollout.rlt_feature_model.openpi.num_steps"),
    "student_c": select(control, "actor.model.num_action_chunks"),
    "action_d": select(control, "actor.model.action_dim"),
    "val_interval": select(control, "runner.val_check_interval"),
    "save_interval": select(control, "runner.save_interval"),
}

for path in (
    "cluster.component_placement", "runner.max_steps", "runner.resume_dir",
    "runner.val_check_interval", "runner.save_interval", "env.train.total_num_envs",
    "env.train.rollout_epoch", "env.eval.total_num_envs", "env.eval.rollout_epoch",
    "actor.global_batch_size", "actor.micro_batch_size", "actor.seed",
    "algorithm.update_epoch", "algorithm.critic_actor_ratio",
    "algorithm.rlt_schedule", "algorithm.replay_buffer", "algorithm.actor_weight_schedule",
    "actor.optim", "actor.critic_optim", "rollout.rlt_feature_model.model_path",
):
    if select(control, path) != select(dvac, path):
        raise SystemExit(f"unexpected A/B mismatch at {path}")

dvac_cfg = select(dvac, "algorithm.rlt_dvac")
assert "rlt_dvac" not in control["algorithm"]
assert dvac_cfg["mode"] == "apply"
assert float(dvac_cfg["z_clip"]) == 2.0
assert float(dvac_cfg["strength"]) == 0.25
assert int(dvac_cfg["selected_l"]) == 3
assert int(dvac_cfg["applied_horizon"]) == 10
assert control["rollout"]["rlt_feature_model"]["openpi"].get("rlt_dvac_mode", "off") == "off"
assert select(dvac, "rollout.rlt_feature_model.openpi.rlt_dvac_mode") == "apply"

payload = {
    "status": "SINGLE_GPU_AB_CONTRACT_OK",
    "common": common,
    "derived_gradient_accumulation_world1": common["global_batch"] // common["micro_batch"],
    "expected_method_differences": {
        "control_dvac_mode": "off",
        "dvac_mode": dvac_cfg["mode"],
        "z_clip": dvac_cfg["z_clip"],
        "strength": dvac_cfg["strength"],
        "weight_range": [1 - dvac_cfg["z_clip"] * dvac_cfg["strength"], 1 + dvac_cfg["z_clip"] * dvac_cfg["strength"]],
    },
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

sha256sum "$out/control.resolved.yaml" "$out/dvac.resolved.yaml" >"$out/RESOLVED_SHA256.txt"

git add \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_fresh480_control.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_single_gpu_teacher_dvac_w05to15_fresh480.yaml
git diff --cached --check
git diff --cached --stat
git commit -m 'config(rlt): add paired single GPU 480 runs'
git push origin codex/rlt-teacher-dvac-weighting

git rev-parse HEAD | tee "$out/source_head.txt"
git status --short
