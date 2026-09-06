#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=a5b94b6f10a9212502d6930f07543f61e31af52e
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
NAME=dvac-action-adv-w0to2-smoke2-2gpu64x4-b1024-noeval-phys23-v1
RUN="$ROOT/runs/$NAME"
PACKET="$ROOT/packets/$NAME"
REFERENCE="$ROOT/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2/runtime/resolved.yaml"

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test -s "$REFERENCE"
test ! -e "$RUN"
test ! -e "$PACKET"
install -d -m 755 "$PACKET"

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

compose() {
  local output=$1 logprob=$2 mode=$3 application=$4 min=$5 max=$6
  local -a args=(
    --config-path "$WT/examples/embodiment/config"
    --config-name robotwin_adjust_bottle_grpo_openpi
    'cluster.component_placement={actor\, env\, rollout:"2,3"}'
    "runner.logger.log_path=$RUN"
    runner.logger.experiment_name=robotwin_dvac_action_adv_w0to2_smoke2_2gpu64x4_b1024_noeval_phys23_v1
    runner.max_epochs=1000
    runner.max_steps=2
    runner.val_check_interval=-1
    runner.save_interval=10
    runner.resume_dir=null
    algorithm.update_epoch=2
    algorithm.adv_type=grpo
    algorithm.filter_rewards=true
    "algorithm.logprob_type=$logprob"
    "algorithm.dvac_gradient_weighting.mode=$mode"
    "algorithm.dvac_gradient_weighting.application=$application"
    algorithm.dvac_gradient_weighting.selected_l=3
    algorithm.dvac_gradient_weighting.warmup_steps=1
    algorithm.dvac_gradient_weighting.window_steps=5
    "algorithm.dvac_gradient_weighting.weight_min=$min"
    "algorithm.dvac_gradient_weighting.weight_max=$max"
    env.train.total_num_envs=64
    env.train.rollout_epoch=4
    env.train.max_episode_steps=200
    env.train.max_steps_per_rollout_epoch=200
    "env.train.assets_path=$ROBOTWIN"
    "env.train.video_cfg.video_base_dir=$RUN/video/train"
    "env.train.task_config.save_path=$RUN/robotwin_data/train"
    env.eval.total_num_envs=32
    env.eval.rollout_epoch=1
    env.eval.max_episode_steps=200
    env.eval.max_steps_per_rollout_epoch=200
    env.eval.use_fixed_reset_state_ids=true
    "env.eval.assets_path=$ROBOTWIN"
    "env.eval.video_cfg.video_base_dir=$RUN/video/eval"
    "env.eval.task_config.save_path=$RUN/robotwin_data/eval"
    actor.micro_batch_size=32
    actor.global_batch_size=1024
    "actor.model.model_path=$MODEL"
  )
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" --cfg job --resolve > "$output"
  if [[ "$mode" == apply ]]; then
    printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" > "$PACKET/command.txt"
    printf '\n' >> "$PACKET/command.txt"
  fi
}

compose "$PACKET/control_same_code_resolved.yaml" chunk_level off logprob_st null null
compose "$PACKET/resolved.yaml" action_level apply action_advantage 0.0 2.0

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
    "runner.max_steps", "runner.val_check_interval",
    "env.train.seeds_path", "env.eval.seeds_path",
    "env.train.video_cfg.video_base_dir", "env.train.task_config.save_path",
    "env.eval.video_cfg.video_base_dir", "env.eval.task_config.save_path",
    "actor.model.output_dir", "algorithm.dvac_gradient_weighting.output_dir",
    "algorithm.dvac_gradient_weighting.application",
}
assert not sorted(set(reference_diff) - allowed_reference), sorted(set(reference_diff) - allowed_reference)

method_diff = difference(control, method)
allowed_method = {
    "algorithm.logprob_type",
    "algorithm.dvac_gradient_weighting.mode",
    "algorithm.dvac_gradient_weighting.application",
    "algorithm.dvac_gradient_weighting.weight_min",
    "algorithm.dvac_gradient_weighting.weight_max",
}
assert set(method_diff) == allowed_method, method_diff

cfg = method
assert cfg["cluster"]["component_placement"] == {"actor, env, rollout": "2,3"}
assert cfg["runner"]["max_steps"] == 2 and cfg["runner"]["val_check_interval"] == -1
assert cfg["algorithm"]["adv_type"] == "grpo" and cfg["algorithm"]["filter_rewards"] is True
assert cfg["algorithm"]["reward_type"] == "chunk_level"
assert cfg["algorithm"]["logprob_type"] == "action_level"
dvac = cfg["algorithm"]["dvac_gradient_weighting"]
assert dvac["mode"] == "apply" and dvac["application"] == "action_advantage"
assert dvac["selected_l"] == 3 and dvac["warmup_steps"] == 1 and dvac["window_steps"] == 5
assert dvac["weight_min"] == 0.0 and dvac["weight_max"] == 2.0
assert cfg["algorithm"]["group_size"] == 8 and cfg["algorithm"]["update_epoch"] == 2
assert cfg["env"]["train"]["total_num_envs"] == 64 and cfg["env"]["train"]["rollout_epoch"] == 4
assert cfg["actor"]["global_batch_size"] == 1024 and cfg["actor"]["micro_batch_size"] == 32

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump({
        "source_head": "a5b94b6f10a9212502d6930f07543f61e31af52e",
        "reference": reference_path,
        "reference_diff": reference_diff,
        "method_diff": method_diff,
        "budget": {
            "physical_gpus": [2, 3], "outer_steps": 2,
            "train_envs": 64, "rollout_epochs": 4, "trajectories_per_step": 256,
            "group_size": 8, "groups_per_step": 32, "max_chunk_records": 1024,
            "global_batch": 1024, "micro_batch": 32, "update_epochs": 2,
            "inline_eval": "disabled for smoke", "terminal_checkpoint": "global_step_2",
        },
    }, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
print("SZ_ACTION_ADV_PACKET_PARITY_OK")
PY

cat > "$PACKET/contract.json" <<EOF
{
  "source_head": "$HEAD",
  "run": "$RUN",
  "physical_gpus": [2, 3],
  "train": "64 env x 4 rollout epochs = 256 trajectories/step; G8; max 1024 chunk records",
  "actor": "GB1024/MB32/update2",
  "method": "trajectory GRPO A_i; action-level ratio/clip; A_eff[i,h]=A_i*stopgrad(raw DVAC w[i,h] in [0,2])",
  "smoke": "2 outer steps; step1 warm-up all-one; step2 must be non-uniform; inline eval disabled",
  "normal_stop": "complete global_step_2 checkpoint and exit0",
  "hard_timeout_seconds": 7200
}
EOF

sha256sum "$PACKET/resolved.yaml" "$PACKET/control_same_code_resolved.yaml" "$PACKET/command.txt"
cat "$PACKET/contract.json"
echo SZ_ACTION_ADV_SMOKE2_PACKET_READY
