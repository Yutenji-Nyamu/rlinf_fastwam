#!/usr/bin/env bash
set -euo pipefail

# ACTION_FIX_HEAD is filled after the fix commit is created and pushed.
ACTION_FIX_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-dvac-action-adv-fix
ACTION_FIX_HEAD=e434f409b21d281ce883df29487ecae7cb3e4839
ST_WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
ST_HEAD=0e28ac6f09f821ea12e7d54eba7118ce0000ca86
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
REFERENCE="$ROOT/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2/runtime/resolved.yaml"
ACTION_NAME=dvac-action-adv-fix-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1
ST_NAME=dvac-st-global-z-w0p5to1p5-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v1
ACTION_RUN="$ROOT/runs/$ACTION_NAME"
ST_RUN="$ROOT/runs/$ST_NAME"
ACTION_PACKET="$ROOT/packets/$ACTION_NAME"
ST_PACKET="$ROOT/packets/$ST_NAME"

test "$ACTION_FIX_HEAD" != __ACTION_FIX_HEAD__
test "$(git -C "$ACTION_FIX_WT" rev-parse HEAD)" = "$ACTION_FIX_HEAD"
test -z "$(git -C "$ACTION_FIX_WT" status --short)"
test "$(git -C "$ST_WT" rev-parse HEAD)" = "$ST_HEAD"
test -z "$(git -C "$ST_WT" status --short)"
test -s "$REFERENCE"
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
test -d "$ROBOTWIN"
for run in "$ACTION_RUN" "$ST_RUN"; do test ! -e "$run"; done
for packet in "$ACTION_PACKET" "$ST_PACKET"; do
  if [[ -e "$packet" ]]; then test ! -e "$packet/packet_complete.txt"; fi
  install -d -m 755 "$packet"
done

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
BASE_PYTHONPATH=${PYTHONPATH:-}

use_source() {
  local wt=$1
  export REPO_PATH="$wt" EMBODIED_PATH="$wt/examples/embodiment" RLINF_CODE_WORKING_DIR="$wt"
  export PYTHONPATH="$wt:$ROBOTWIN${BASE_PYTHONPATH:+:$BASE_PYTHONPATH}"
}

compose_action() {
  local output=$1 logprob=$2 mode=$3 application=$4 min=$5 max=$6 experiment=$7
  local -a args=(
    --config-path "$ACTION_FIX_WT/examples/embodiment/config"
    --config-name robotwin_adjust_bottle_grpo_openpi
    'cluster.component_placement={actor\, env\, rollout:"4,5"}'
    "runner.logger.log_path=$ACTION_RUN"
    "runner.logger.experiment_name=$experiment"
    runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
    algorithm.update_epoch=2 algorithm.adv_type=grpo algorithm.filter_rewards=true
    "algorithm.logprob_type=$logprob"
    "algorithm.dvac_gradient_weighting.mode=$mode"
    "algorithm.dvac_gradient_weighting.application=$application"
    algorithm.dvac_gradient_weighting.selected_l=3
    algorithm.dvac_gradient_weighting.warmup_steps=1
    algorithm.dvac_gradient_weighting.window_steps=5
    "algorithm.dvac_gradient_weighting.weight_min=$min"
    "algorithm.dvac_gradient_weighting.weight_max=$max"
    env.train.total_num_envs=64 env.train.rollout_epoch=4
    env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
    "env.train.assets_path=$ROBOTWIN"
    env.train.video_cfg.save_video=true
    "env.train.video_cfg.video_base_dir=$ACTION_RUN/video/train"
    "env.train.task_config.save_path=$ACTION_RUN/robotwin_data/train"
    env.eval.total_num_envs=32 env.eval.rollout_epoch=1
    env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
    env.eval.use_fixed_reset_state_ids=true
    "env.eval.assets_path=$ROBOTWIN"
    env.eval.video_cfg.save_video=true
    "env.eval.video_cfg.video_base_dir=$ACTION_RUN/video/eval"
    "env.eval.task_config.save_path=$ACTION_RUN/robotwin_data/eval"
    actor.micro_batch_size=32 actor.global_batch_size=1024
    "actor.model.model_path=$MODEL"
  )
  "$VENV/bin/python" "$ACTION_FIX_WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" --cfg job --resolve > "$output"
  if [[ "$mode" == apply ]]; then
    printf '%q ' "$VENV/bin/python" "$ACTION_FIX_WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" > "$ACTION_PACKET/command.txt"
    printf '\n' >> "$ACTION_PACKET/command.txt"
  fi
}

compose_st() {
  local output=$1 mode=$2 min=$3 max=$4 experiment=$5
  local -a args=(
    --config-path "$ST_WT/examples/embodiment/config"
    --config-name robotwin_adjust_bottle_grpo_openpi
    'cluster.component_placement={actor\, env\, rollout:"6,7"}'
    "runner.logger.log_path=$ST_RUN"
    "runner.logger.experiment_name=$experiment"
    runner.max_epochs=1000 runner.max_steps=100 runner.val_check_interval=5 runner.save_interval=10 runner.resume_dir=null
    algorithm.update_epoch=2 algorithm.adv_type=grpo algorithm.filter_rewards=true
    algorithm.logprob_type=chunk_level
    "algorithm.dvac_gradient_weighting.mode=$mode"
    algorithm.dvac_gradient_weighting.selected_l=3
    algorithm.dvac_gradient_weighting.warmup_steps=1
    algorithm.dvac_gradient_weighting.window_steps=5
    "algorithm.dvac_gradient_weighting.weight_min=$min"
    "algorithm.dvac_gradient_weighting.weight_max=$max"
    env.train.total_num_envs=64 env.train.rollout_epoch=4
    env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200
    "env.train.assets_path=$ROBOTWIN"
    env.train.video_cfg.save_video=true
    "env.train.video_cfg.video_base_dir=$ST_RUN/video/train"
    "env.train.task_config.save_path=$ST_RUN/robotwin_data/train"
    env.eval.total_num_envs=32 env.eval.rollout_epoch=1
    env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200
    env.eval.use_fixed_reset_state_ids=true
    "env.eval.assets_path=$ROBOTWIN"
    env.eval.video_cfg.save_video=true
    "env.eval.video_cfg.video_base_dir=$ST_RUN/video/eval"
    "env.eval.task_config.save_path=$ST_RUN/robotwin_data/eval"
    actor.micro_batch_size=32 actor.global_batch_size=1024
    "actor.model.model_path=$MODEL"
  )
  "$VENV/bin/python" "$ST_WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" --cfg job --resolve > "$output"
  if [[ "$mode" == apply ]]; then
    printf '%q ' "$VENV/bin/python" "$ST_WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" > "$ST_PACKET/command.txt"
    printf '\n' >> "$ST_PACKET/command.txt"
  fi
}

use_source "$ACTION_FIX_WT"
compose_action "$ACTION_PACKET/control_same_code_resolved.yaml" chunk_level off logprob_st null null robotwin_action_adv_fix_same_code_control_not_run
compose_action "$ACTION_PACKET/resolved.yaml" action_level apply action_advantage 0.0 2.0 robotwin_dvac_action_adv_fix_w0to2_formal100_2gpu64x4_b1024_fixed32_eval5_phys45_v1

use_source "$ST_WT"
compose_st "$ST_PACKET/control_same_code_resolved.yaml" off null null robotwin_st_half_same_code_control_not_run
compose_st "$ST_PACKET/resolved.yaml" apply 0.5 1.5 robotwin_dvac_st_global_z_w0p5to1p5_formal100_2gpu64x4_b1024_fixed32_eval5_phys67_v1

"$VENV/bin/python" - \
  "$REFERENCE" \
  "$ACTION_PACKET/control_same_code_resolved.yaml" "$ACTION_PACKET/resolved.yaml" "$ACTION_PACKET/parity.json" "$ACTION_FIX_HEAD" \
  "$ST_PACKET/control_same_code_resolved.yaml" "$ST_PACKET/resolved.yaml" "$ST_PACKET/parity.json" "$ST_HEAD" <<'PY'
from __future__ import annotations
import json, sys, yaml

(reference_path,
 action_control_path, action_path, action_output, action_head,
 st_control_path, st_path, st_output, st_head) = sys.argv[1:]

def load(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)

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

def check_budget(cfg, placement, logprob, mode, minimum, maximum):
    assert cfg["cluster"]["component_placement"] == {"actor, env, rollout": placement}
    assert cfg["runner"]["max_steps"] == 100 and cfg["runner"]["resume_dir"] is None
    assert cfg["runner"]["val_check_interval"] == 5 and cfg["runner"]["save_interval"] == 10
    assert cfg["algorithm"]["adv_type"] == "grpo" and cfg["algorithm"]["filter_rewards"] is True
    assert cfg["algorithm"]["reward_type"] == "chunk_level"
    assert cfg["algorithm"]["logprob_type"] == logprob
    assert cfg["algorithm"]["group_size"] == 8 and cfg["algorithm"]["update_epoch"] == 2
    weight = cfg["algorithm"]["dvac_gradient_weighting"]
    assert weight["mode"] == mode and weight["selected_l"] == 3
    assert weight["warmup_steps"] == 1 and weight["window_steps"] == 5
    assert weight["weight_min"] == minimum and weight["weight_max"] == maximum
    assert cfg["env"]["train"]["total_num_envs"] == 64
    assert cfg["env"]["train"]["rollout_epoch"] == 4
    assert cfg["env"]["eval"]["total_num_envs"] == 32
    assert cfg["env"]["eval"]["rollout_epoch"] == 1
    assert cfg["env"]["eval"]["use_fixed_reset_state_ids"] is True
    assert cfg["actor"]["global_batch_size"] == 1024
    assert cfg["actor"]["micro_batch_size"] == 32

reference = load(reference_path)
action_control, action = load(action_control_path), load(action_path)
st_control, st = load(st_control_path), load(st_path)

run_scoped = {
    "runner.logger.log_path", "runner.logger.experiment_name",
    "env.train.seeds_path", "env.eval.seeds_path",
    "env.train.video_cfg.video_base_dir", "env.train.task_config.save_path",
    "env.eval.video_cfg.video_base_dir", "env.eval.task_config.save_path",
    "actor.model.output_dir", "algorithm.dvac_gradient_weighting.output_dir",
}

action_reference_diff = difference(reference, action_control)
allowed_action_reference = run_scoped | {"algorithm.dvac_gradient_weighting.application"}
unexpected_action_reference = sorted(set(action_reference_diff) - allowed_action_reference)
assert not unexpected_action_reference, unexpected_action_reference
action_method_diff = difference(action_control, action)
allowed_action_method = {
    "runner.logger.experiment_name", "algorithm.logprob_type",
    "algorithm.dvac_gradient_weighting.mode", "algorithm.dvac_gradient_weighting.application",
    "algorithm.dvac_gradient_weighting.output_dir",
    "algorithm.dvac_gradient_weighting.weight_min", "algorithm.dvac_gradient_weighting.weight_max",
}
assert set(action_method_diff) == allowed_action_method, action_method_diff
check_budget(action, "4,5", "action_level", "apply", 0.0, 2.0)
assert action["algorithm"]["dvac_gradient_weighting"]["application"] == "action_advantage"

st_reference_diff = difference(reference, st_control)
allowed_st_reference = run_scoped | {"cluster.component_placement.actor, env, rollout"}
unexpected_st_reference = sorted(set(st_reference_diff) - allowed_st_reference)
assert not unexpected_st_reference, unexpected_st_reference
st_method_diff = difference(st_control, st)
allowed_st_method = {
    "runner.logger.experiment_name", "algorithm.dvac_gradient_weighting.mode",
    "algorithm.dvac_gradient_weighting.output_dir",
    "algorithm.dvac_gradient_weighting.weight_min", "algorithm.dvac_gradient_weighting.weight_max",
}
assert set(st_method_diff) == allowed_st_method, st_method_diff
check_budget(st, "6,7", "chunk_level", "apply", 0.5, 1.5)

common_budget = {
    "physical_gpus": 2, "train_envs": 64, "train_envs_per_gpu": 32,
    "rollout_epochs": 4, "trajectories_per_step": 256,
    "group_size": 8, "groups_per_step": 32, "max_chunk_records": 1024,
    "global_batch": 1024, "micro_batch": 32, "update_epochs": 2,
    "inline_eval": "fixed32 every 5 steps", "checkpoint": "default DCP every 10 steps",
}
outputs = (
    (action_output, {
        "source_head": action_head, "reference": reference_path,
        "reference_diff": action_reference_diff, "unexpected_reference_diff": unexpected_action_reference,
        "method_diff": action_method_diff, "budget": common_budget,
        "method": "GRPO DVAC Action-Adv Fix [0,2]: action-level ratio/clip; A_eff=A*w; sum valid H then mean queries",
    }),
    (st_output, {
        "source_head": st_head, "reference": reference_path,
        "reference_diff": st_reference_diff, "unexpected_reference_diff": unexpected_st_reference,
        "method_diff": st_method_diff, "budget": common_budget,
        "method": "GRPO DVAC ST global-z [0.5,1.5]: Control joint-chunk forward/clip; local backward scaling only",
    }),
)
for path, payload in outputs:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
print("SZ_ACTION_ADV_FIX_AND_ST_HALF_PACKET_PARITY_OK unexpected=0")
PY

cat > "$ACTION_PACKET/contract.json" <<EOF
{
  "source_head": "$ACTION_FIX_HEAD",
  "run": "$ACTION_RUN",
  "physical_gpus": [4, 5],
  "train": "64 env x 4 rollout epochs = 256 trajectories/step; G8; max 1024 chunk records",
  "actor": "GB1024/MB32/update2",
  "method": "trajectory GRPO A; action-level ratio/clip; A_eff=A*stopgrad(DVAC w in [0,2]); fixed source sums valid H then means queries",
  "formal": "fresh 100 steps; fixed32 every5; default DCP checkpoint every10",
  "normal_stop": "complete global_step_100 and exit0",
  "hard_timeout_seconds": 216000
}
EOF

cat > "$ST_PACKET/contract.json" <<EOF
{
  "source_head": "$ST_HEAD",
  "run": "$ST_RUN",
  "physical_gpus": [6, 7],
  "train": "64 env x 4 rollout epochs = 256 trajectories/step; G8; max 1024 chunk records",
  "actor": "GB1024/MB32/update2",
  "method": "trajectory GRPO A; Control joint-chunk ratio/clip; DVAC ST global-z local gradient multipliers in [0.5,1.5]",
  "formal": "fresh 100 steps; fixed32 every5; default DCP checkpoint every10",
  "normal_stop": "complete global_step_100 and exit0",
  "hard_timeout_seconds": 216000
}
EOF

sha256sum \
  "$ACTION_PACKET/resolved.yaml" "$ACTION_PACKET/control_same_code_resolved.yaml" "$ACTION_PACKET/command.txt" \
  "$ST_PACKET/resolved.yaml" "$ST_PACKET/control_same_code_resolved.yaml" "$ST_PACKET/command.txt"
cat "$ACTION_PACKET/parity.json"
cat "$ST_PACKET/parity.json"
printf '%s\n' "completed_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" > "$ACTION_PACKET/packet_complete.txt"
printf '%s\n' "completed_at=$(TZ=Asia/Shanghai date --iso-8601=seconds)" > "$ST_PACKET/packet_complete.txt"
echo SZ_ACTION_ADV_FIX_AND_ST_HALF_DUAL_FORMAL100_PACKETS_READY
