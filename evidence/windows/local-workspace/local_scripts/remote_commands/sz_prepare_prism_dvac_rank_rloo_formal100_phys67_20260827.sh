#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/prism-dvac-rank-rloo
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=3f977ce7f0c9271a4e164b4d602166286a9ad9f3
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
NAME=prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
RUN="$ROOT/runs/$NAME"
PACKET="$ROOT/packets/$NAME"
REFERENCE="$ROOT/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2/runtime/resolved.yaml"

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test -s "$REFERENCE"
test ! -e "$RUN"
test ! -e "$PACKET"
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
install -d -m 755 "$PACKET"

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

compose() {
  local output=$1 adv=$2 filter=$3 enabled=$4
  local -a args=(
    --config-path "$WT/examples/embodiment/config"
    --config-name robotwin_adjust_bottle_grpo_openpi
    'cluster.component_placement={actor\, env\, rollout:"6,7"}'
    "runner.logger.log_path=$RUN"
    runner.logger.experiment_name=robotwin_prism_dvac_rank_rloo_formal100_2gpu64x4_b1024_fixed32_eval5_phys67_v1
    runner.max_epochs=1000
    runner.max_steps=100
    runner.val_check_interval=5
    runner.save_interval=10
    runner.resume_dir=null
    algorithm.update_epoch=2
    "algorithm.adv_type=$adv"
    "algorithm.filter_rewards=$filter"
    algorithm.dvac_gradient_weighting.mode=off
    algorithm.dvac_gradient_weighting.weight_min=null
    algorithm.dvac_gradient_weighting.weight_max=null
    "algorithm.prism_dvac.enabled=$enabled"
    algorithm.prism_dvac.selected_l=3
    algorithm.prism_dvac.quality_lambda=0.2
    algorithm.prism_dvac.log_eps=1.0e-12
    env.train.total_num_envs=64
    env.train.rollout_epoch=4
    env.train.max_episode_steps=200
    env.train.max_steps_per_rollout_epoch=200
    "env.train.assets_path=$ROBOTWIN"
    env.train.video_cfg.save_video=true
    "env.train.video_cfg.video_base_dir=$RUN/video/train"
    "env.train.task_config.save_path=$RUN/robotwin_data/train"
    env.eval.total_num_envs=32
    env.eval.rollout_epoch=1
    env.eval.max_episode_steps=200
    env.eval.max_steps_per_rollout_epoch=200
    env.eval.use_fixed_reset_state_ids=true
    "env.eval.assets_path=$ROBOTWIN"
    env.eval.video_cfg.save_video=true
    "env.eval.video_cfg.video_base_dir=$RUN/video/eval"
    "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
    actor.micro_batch_size=32
    actor.global_batch_size=1024
    "actor.model.model_path=$MODEL"
  )
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" --cfg job --resolve > "$output"
  if [[ "$enabled" == true ]]; then
    printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" > "$PACKET/command.txt"
    printf '\n' >> "$PACKET/command.txt"
  fi
}

compose "$PACKET/control_same_code_resolved.yaml" grpo true false
compose "$PACKET/resolved.yaml" prism_rloo false true

"$VENV/bin/python" - "$REFERENCE" "$PACKET/control_same_code_resolved.yaml" "$PACKET/resolved.yaml" "$PACKET/parity.json" <<'PY'
from __future__ import annotations
import json, sys, yaml

reference_path, control_path, method_path, output_path = sys.argv[1:]
reference = yaml.safe_load(open(reference_path, encoding="utf-8"))
control = yaml.safe_load(open(control_path, encoding="utf-8"))
method = yaml.safe_load(open(method_path, encoding="utf-8"))

def flat(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(flat(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        out[prefix] = value
    else:
        out[prefix] = value
    return out

def difference(a, b):
    left, right, missing = flat(a), flat(b), object()
    return {key: {"left": left.get(key, "<MISSING>"), "right": right.get(key, "<MISSING>")}
            for key in sorted(set(left) | set(right)) if left.get(key, missing) != right.get(key, missing)}

reference_diff = difference(reference, control)
allowed_reference = {
    "cluster.component_placement.actor, env, rollout",
    "runner.logger.log_path", "runner.logger.experiment_name",
    "env.train.seeds_path", "env.eval.seeds_path",
    "env.train.video_cfg.video_base_dir", "env.train.task_config.save_path",
    "env.eval.video_cfg.video_base_dir", "env.eval.task_config.save_path",
    "actor.model.output_dir", "algorithm.dvac_gradient_weighting.output_dir",
    "algorithm.prism_dvac.enabled", "algorithm.prism_dvac.selected_l",
    "algorithm.prism_dvac.quality_lambda", "algorithm.prism_dvac.log_eps",
}
unexpected_reference = sorted(set(reference_diff) - allowed_reference)
assert not unexpected_reference, unexpected_reference

method_diff = difference(control, method)
allowed_method = {
    "algorithm.adv_type",
    "algorithm.filter_rewards",
    "algorithm.prism_dvac.enabled",
}
unexpected_method = sorted(set(method_diff) - allowed_method)
assert not unexpected_method, unexpected_method
assert set(method_diff) == allowed_method, method_diff

cfg = method
assert cfg["cluster"]["component_placement"] == {"actor, env, rollout": "6,7"}
assert cfg["runner"]["max_steps"] == 100
assert cfg["runner"]["val_check_interval"] == 5
assert cfg["runner"]["save_interval"] == 10
assert cfg["runner"]["resume_dir"] is None
assert cfg["algorithm"]["adv_type"] == "prism_rloo"
assert cfg["algorithm"]["filter_rewards"] is False
assert cfg["algorithm"]["dvac_gradient_weighting"]["mode"] == "off"
assert cfg["algorithm"]["prism_dvac"] == {
    "enabled": True, "selected_l": 3, "quality_lambda": 0.2, "log_eps": 1e-12,
}
assert cfg["algorithm"]["group_size"] == 8
assert cfg["algorithm"]["update_epoch"] == 2
assert cfg["env"]["train"]["total_num_envs"] == 64
assert cfg["env"]["train"]["rollout_epoch"] == 4
assert cfg["env"]["train"]["auto_reset"] is False
assert cfg["env"]["train"]["ignore_terminations"] is False
assert cfg["env"]["train"]["use_custom_reward"] is True
assert cfg["env"]["train"]["use_rel_reward"] is True
assert cfg["env"]["train"]["reward_coef"] == 1.0
assert cfg["env"]["eval"]["total_num_envs"] == 32
assert cfg["env"]["eval"]["rollout_epoch"] == 1
assert cfg["env"]["eval"]["use_fixed_reset_state_ids"] is True
assert cfg["actor"]["global_batch_size"] == 1024
assert cfg["actor"]["micro_batch_size"] == 32

payload = {
    "source_head": "3f977ce7f0c9271a4e164b4d602166286a9ad9f3",
    "reference": reference_path,
    "same_code_control": control_path,
    "method": method_path,
    "reference_diff": reference_diff,
    "unexpected_reference_diff": unexpected_reference,
    "method_diff": method_diff,
    "unexpected_method_diff": unexpected_method,
    "budget": {
        "physical_gpus": [6, 7], "outer_steps": 100,
        "train_envs": 64, "rollout_epochs": 4, "trajectories": 256,
        "group_size": 8, "groups": 32, "max_chunk_records": 1024,
        "global_batch": 1024, "micro_batch": 32, "update_epoch": 2,
        "optimizer_calls_per_step_per_rank": 2,
        "fixed_eval": "32 x 1 every 5 steps", "checkpoint": "every 10 steps",
    },
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
print("SZ_PRISM_FORMAL_PACKET_PARITY_OK")
PY

test "$(sha256sum "$WT/rlinf/envs/robotwin/seeds/train_seeds.json" | cut -d' ' -f1)" = \
     "$(sha256sum /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current/rlinf/envs/robotwin/seeds/train_seeds.json | cut -d' ' -f1)"
test "$(sha256sum "$WT/rlinf/envs/robotwin/seeds/eval_seeds.json" | cut -d' ' -f1)" = \
     "$(sha256sum /data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current/rlinf/envs/robotwin/seeds/eval_seeds.json | cut -d' ' -f1)"

cat > "$PACKET/contract.json" <<EOF
{
  "source_head": "$HEAD",
  "run": "$RUN",
  "physical_gpus": [6, 7],
  "outer_steps": 100,
  "train": "64 env x 4 rollout epochs = 256 trajectories; G8 = 32 groups; max 1024 chunk records",
  "actor": "GB1024/MB32/update2; 512 records/rank; 2 optimizer calls/rank",
  "method": "trajectory mean log V_L3; within-G8 lower-is-better average rank; reward=success+0.2q; sibling-mean RLOO; no std; DVAC-ST off",
  "eval": "fixed32 as 32 env x 1 wave every 5 steps",
  "checkpoint": "every 10 steps",
  "normal_stop": "complete step100, final checkpoint, exit0",
  "hard_timeout_seconds": 216000
}
EOF

sha256sum "$PACKET/resolved.yaml" "$PACKET/control_same_code_resolved.yaml" "$PACKET/command.txt"
cat "$PACKET/contract.json"
cat "$PACKET/parity.json"
echo SZ_PRISM_DVAC_RANK_RLOO_FORMAL100_PACKET_READY
