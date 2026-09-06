#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
HEAD=0e28ac6f09f821ea12e7d54eba7118ce0000ca86
ROOT=/data/chenyiteng/results/rlinf-shenzhen/grpo
CONTROL=grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2
DVAC=dvac-global-z-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v2

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --short)"
test -s "$MODEL/model-00001-of-00002.safetensors"
test -s "$MODEL/model-00002-of-00002.safetensors"
test -d "$ROBOTWIN"
for name in "$CONTROL" "$DVAC"; do
  test ! -e "$ROOT/runs/$name"
  test ! -e "$ROOT/packets/$name"
  install -d -m 755 "$ROOT/packets/$name"
done

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

compose() {
  local name=$1 placement=$2 experiment=$3 mode=$4
  local run="$ROOT/runs/$name" packet="$ROOT/packets/$name"
  local -a args=(
    --config-path "$WT/examples/embodiment/config"
    --config-name robotwin_adjust_bottle_grpo_openpi
    "cluster.component_placement={actor\\, env\\, rollout:\"$placement\"}"
    "runner.logger.log_path=$run"
    "runner.logger.experiment_name=$experiment"
    runner.max_epochs=1000
    runner.max_steps=100
    runner.val_check_interval=5
    runner.save_interval=10
    runner.resume_dir=null
    algorithm.update_epoch=2
    "algorithm.dvac_gradient_weighting.mode=$mode"
    algorithm.dvac_gradient_weighting.weight_min=null
    algorithm.dvac_gradient_weighting.weight_max=null
    env.train.total_num_envs=64
    env.train.rollout_epoch=4
    env.train.max_episode_steps=200
    env.train.max_steps_per_rollout_epoch=200
    "env.train.assets_path=$ROBOTWIN"
    env.train.video_cfg.save_video=true
    "env.train.video_cfg.video_base_dir=$run/video/train"
    "env.train.task_config.save_path=$run/robotwin_data/train"
    env.eval.total_num_envs=32
    env.eval.rollout_epoch=1
    env.eval.max_episode_steps=200
    env.eval.max_steps_per_rollout_epoch=200
    env.eval.use_fixed_reset_state_ids=true
    "env.eval.assets_path=$ROBOTWIN"
    env.eval.video_cfg.save_video=true
    "env.eval.video_cfg.video_base_dir=$run/video/eval"
    "env.eval.task_config.save_path=$run/robotwin_data/eval"
    actor.micro_batch_size=32
    actor.global_batch_size=1024
    "actor.model.model_path=$MODEL"
  )
  "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" --cfg job --resolve > "$packet/resolved.yaml"
  printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${args[@]}" > "$packet/command.txt"
  printf '\n' >> "$packet/command.txt"
}

compose "$CONTROL" '4,5' robotwin_grpo_control_2gpu64x4_b1024_fixed32_eval5_v2 off
compose "$DVAC" '6,7' robotwin_grpo_dvac_global_z_w0to2_2gpu64x4_b1024_fixed32_eval5_v2 apply

"$VENV/bin/python" - "$ROOT/packets/$CONTROL/resolved.yaml" "$ROOT/packets/$DVAC/resolved.yaml" "$ROOT/packets/$CONTROL/pair_parity.json" <<'PY'
from __future__ import annotations
import json, sys, yaml

control_path, dvac_path, output_path = sys.argv[1:]
control = yaml.safe_load(open(control_path, encoding="utf-8"))
dvac = yaml.safe_load(open(dvac_path, encoding="utf-8"))

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

a, b = flat(control), flat(dvac)
missing = object()
diff = {key: {"control": a.get(key, "<MISSING>"), "dvac": b.get(key, "<MISSING>")}
        for key in sorted(set(a) | set(b)) if a.get(key, missing) != b.get(key, missing)}
allowed = {
    "cluster.component_placement.actor, env, rollout",
    "runner.logger.log_path", "runner.logger.experiment_name",
    "env.train.video_cfg.video_base_dir", "env.train.task_config.save_path",
    "env.eval.video_cfg.video_base_dir", "env.eval.task_config.save_path",
    "actor.model.output_dir", "algorithm.dvac_gradient_weighting.output_dir",
    "algorithm.dvac_gradient_weighting.mode",
}
unexpected = sorted(set(diff) - allowed)
assert not unexpected, unexpected

for cfg, mode, placement in ((control, "off", "4,5"), (dvac, "apply", "6,7")):
    assert cfg["cluster"]["component_placement"] == {"actor, env, rollout": placement}
    assert cfg["runner"]["resume_dir"] is None
    assert cfg["runner"]["max_steps"] == 100
    assert cfg["runner"]["val_check_interval"] == 5
    assert cfg["runner"]["save_interval"] == 10
    assert cfg["env"]["train"]["total_num_envs"] == 64
    assert cfg["env"]["train"]["rollout_epoch"] == 4
    assert cfg["env"]["eval"]["total_num_envs"] == 32
    assert cfg["env"]["eval"]["rollout_epoch"] == 1
    assert cfg["env"]["eval"]["use_fixed_reset_state_ids"] is True
    assert cfg["algorithm"]["group_size"] == 8
    assert cfg["algorithm"]["update_epoch"] == 2
    assert cfg["actor"]["global_batch_size"] == 1024
    assert cfg["actor"]["micro_batch_size"] == 32
    weight = cfg["algorithm"]["dvac_gradient_weighting"]
    assert weight["mode"] == mode
    assert weight["selected_l"] == 3
    assert weight["window_steps"] == 5 and weight["warmup_steps"] == 1
    assert weight["z_clip"] == 2.0 and weight["strength"] == 0.5
    assert weight["weight_min"] is None and weight["weight_max"] is None

payload = {
    "control": control_path,
    "dvac": dvac_path,
    "differences": diff,
    "allowed_differences": sorted(allowed),
    "unexpected_differences": unexpected,
    "budget": {
        "world_size": 2,
        "train_env_per_rank": 32,
        "trajectories_per_step": 256,
        "groups_g8": 32,
        "max_chunk_records": 1024,
        "records_per_rank": 512,
        "global_batch": 1024,
        "micro_batch": 32,
        "optimizer_calls_per_step": 2,
        "fixed_eval": "32 x 1",
    },
    "conclusion": "control and DVAC differ only in method mode, placement, and run-scoped paths",
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
print("DUAL_GRPO_PAIR_PARITY_OK unexpected=0")
PY
cp "$ROOT/packets/$CONTROL/pair_parity.json" "$ROOT/packets/$DVAC/pair_parity.json"

for name in "$CONTROL" "$DVAC"; do
  "$VENV/bin/python" - "$ROOT/packets/$name/resolved.yaml" "$ROOT/packets/$name/contract.json" "$HEAD" <<'PY'
import json, sys, yaml
resolved, output, head = sys.argv[1:]
cfg = yaml.safe_load(open(resolved, encoding="utf-8"))
payload = {
    "source_head": head,
    "placement": cfg["cluster"]["component_placement"]["actor, env, rollout"],
    "method": cfg["algorithm"]["dvac_gradient_weighting"]["mode"],
    "fresh_start": True,
    "outer_steps": 100,
    "train": "64 env x 4 rollout epochs = 256 trajectories/step; G8; 32 groups; max 1024 chunk records",
    "actor": "GB1024/MB32/update2; 512 records/rank; 2 optimizer calls/step",
    "eval": "fixed32 as 32 env x 1 wave every 5 steps",
    "checkpoint": "every 10 steps",
    "normal_stop": "complete step 100",
    "hard_timeout_seconds": 216000,
}
with open(output, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
done

cat "$ROOT/packets/$CONTROL/contract.json"
cat "$ROOT/packets/$DVAC/contract.json"
cat "$ROOT/packets/$CONTROL/pair_parity.json"
echo SZ_DUAL_GRPO_2GPU_FIXED32_FORMAL100_V2_PREPARED



