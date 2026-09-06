#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
NATIVE_VENV=/home/chenyiteng/venvs/lerobot-v060-sidney-py310
NATIVE_SRC=/data/chenyiteng/projects/lerobot-sidney/lerobot-v060-py310-compat
NATIVE_MODEL=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
MODEL=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
HEAD=92fea8f72271b1ff37e58f3433a6fb64abe316cc
ROOT=/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney

P_NAME=parity-m10-phys4-v6-chw
A_NAME=b1-adjust-badseed-retry-m10-phys4-v9
B_NAME=move-grpo-smoke1-2gpu64x1-g8-b512-u2-m10-fixed8-phys45-localshard-v9
P_PACKET="$ROOT/packets/$P_NAME"
P_RUN="$ROOT/smokes/$P_NAME"
A_PACKET="$ROOT/packets/$A_NAME"
B_PACKET="$ROOT/packets/$B_NAME"
A_RUN="$ROOT/smokes/$A_NAME"
B_RUN="$ROOT/smokes/$B_NAME"
A_EXPERIMENT=pi05_sidney_b1_adjust_badseed_retry_m10_phys4_v9
B_EXPERIMENT=pi05_sidney_move_grpo_smoke1_2gpu64x1_g8_b512_u2_m10_fixed8_phys45_localshard_v9
CONFIG=robotwin_move_stapler_pad_grpo_openpi_pi05_sidney

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$MODEL/model.safetensors"
test -s "$MODEL/conversion_manifest.json"
test -s "$MODEL/physical-intelligence/robotwin/norm_stats.json"
test -s "$NATIVE_MODEL/model.safetensors"
test -s "$WT/toolkits/lerobot/sidney_pi05_parity.py"
test ! -e "$P_PACKET"
test ! -e "$P_RUN"
test ! -e "$A_PACKET"
test ! -e "$B_PACKET"
test ! -e "$A_RUN"
test ! -e "$B_RUN"

RAY_ADDRESS=172.17.0.1:6389 "$VENV/bin/ray" status > /tmp/sidney_ray_status_20260903.txt
if nvidia-smi -i 4,5 --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'physical GPUs 4/5 are not idle; packet not prepared' >&2
  exit 20
fi

mkdir -p "$P_PACKET" "$A_PACKET" "$B_PACKET"
cp /tmp/sidney_ray_status_20260903.txt "$A_PACKET/ray-status-before.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > "$A_PACKET/gpu-before.csv"
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo > "$A_PACKET/memory-before.txt"
df -h / /home /data > "$A_PACKET/disk-before.txt"
cp "$A_PACKET/ray-status-before.txt" "$B_PACKET/ray-status-before.txt"
cp "$A_PACKET/gpu-before.csv" "$B_PACKET/gpu-before.csv"
cp "$A_PACKET/memory-before.txt" "$B_PACKET/memory-before.txt"
cp "$A_PACKET/disk-before.txt" "$B_PACKET/disk-before.txt"
cp "$A_PACKET/ray-status-before.txt" "$P_PACKET/ray-status-before.txt"
cp "$A_PACKET/gpu-before.csv" "$P_PACKET/gpu-before.csv"
cp "$A_PACKET/memory-before.txt" "$P_PACKET/memory-before.txt"
cp "$A_PACKET/disk-before.txt" "$P_PACKET/disk-before.txt"

mkdir -p "$P_RUN"
"$VENV/bin/python" "$WT/toolkits/lerobot/sidney_pi05_parity.py" prepare \
  --output "$P_RUN/input.npz" --seed 1234 \
  --prompt 'move the stapler onto the pad'
cat > "$P_PACKET/commands.txt" <<EOF
env -u http_proxy -u HTTP_PROXY -u https_proxy -u HTTPS_PROXY -u all_proxy -u ALL_PROXY CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$NATIVE_SRC/src HF_HOME=/data/chenyiteng/cache/huggingface-sidney HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $NATIVE_VENV/bin/python $WT/toolkits/lerobot/sidney_pi05_parity.py export-native --input $P_RUN/input.npz --model $NATIVE_MODEL --source-revision e49e2ab6c11f07511573b67261bd129e88d0a416 --output $P_RUN/native.pt --device cuda:0 > $P_RUN/native.log 2>&1
env -u http_proxy -u HTTP_PROXY -u https_proxy -u HTTPS_PROXY -u all_proxy -u ALL_PROXY CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$WT:$ROBOTWIN HF_HOME=/data/chenyiteng/cache/huggingface-sidney HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi $VENV/bin/python $WT/toolkits/lerobot/sidney_pi05_parity.py export-rlinf --input $P_RUN/input.npz --model $MODEL --output $P_RUN/rlinf.pt --device cuda:0 > $P_RUN/rlinf.log 2>&1
$VENV/bin/python $WT/toolkits/lerobot/sidney_pi05_parity.py compare --native $P_RUN/native.pt --rlinf $P_RUN/rlinf.pt --rtol 1e-2 --atol 5e-3 > $P_RUN/report.json
EOF
cat > "$P_PACKET/contract.json" <<EOF
{
  "kind": "native LeRobot versus current RLinf semantic action parity",
  "physical_gpus": [4],
  "execution": "sequential; models never co-resident",
  "prompt": "move the stapler onto the pad",
  "input_seed": 1234,
  "noise_shape": [1, 50, 32],
  "H": 50,
  "M": 10,
  "compared": ["three_images", "image_masks", "normalized_state14", "padded_state32", "tokens", "token_mask", "model_actions32", "final_actions14"],
  "sample_rtol": 0.01,
  "sample_atol": 0.005
}
EOF

"$VENV/bin/python" - "$WT/rlinf/envs/robotwin/seeds/eval_seeds.json" "$A_PACKET/eval-seeds-adjust-bad1001.json" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
payload = json.loads(source.read_text())
payload["adjust_bottle"]["success_seeds"] = [1001]
target.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"task": "adjust_bottle", "success_seeds": payload["adjust_bottle"]["success_seeds"]}))
PY

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389
export ROBOTWIN_PATH="$ROBOTWIN" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

A_ARGS=(
  --config-path "$WT/examples/embodiment/config" --config-name "$CONFIG"
  'cluster.component_placement={actor\, env\, rollout:"4"}'
  "runner.logger.log_path=$A_RUN" "runner.logger.experiment_name=$A_EXPERIMENT"
  runner.only_eval=true runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1000 runner.resume_dir=null
  env.eval.total_num_envs=1 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=400 env.eval.max_steps_per_rollout_epoch=400
  env.eval.use_fixed_reset_state_ids=true "env.eval.seeds_path=$A_PACKET/eval-seeds-adjust-bad1001.json"
  "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=true
  "env.eval.video_cfg.video_base_dir=$A_RUN/video/eval"
  "env.eval.task_config.save_path=$A_RUN/robotwin_data/eval"
  env.eval.task_config.task_name=adjust_bottle env.eval.task_config.step_lim=400
  "actor.model.model_path=$MODEL" actor.model.num_steps=10 actor.model.openpi.num_steps=10
  ++actor.fsdp_config.checkpoint_format=local_shard
)

B_ARGS=(
  --config-path "$WT/examples/embodiment/config" --config-name "$CONFIG"
  'cluster.component_placement={actor\, env\, rollout:"4,5"}'
  "runner.logger.log_path=$B_RUN" "runner.logger.experiment_name=$B_EXPERIMENT"
  runner.only_eval=false runner.max_epochs=1000 runner.max_steps=1 runner.val_check_interval=1 runner.save_interval=1 runner.resume_dir=null
  algorithm.group_size=8 algorithm.update_epoch=2 algorithm.adv_type=grpo algorithm.loss_type=actor
  algorithm.logprob_type=chunk_level algorithm.filter_rewards=true algorithm.dvac_gradient_weighting.mode=off
  env.train.total_num_envs=64 env.train.rollout_epoch=1
  env.train.max_episode_steps=400 env.train.max_steps_per_rollout_epoch=400
  "env.train.assets_path=$ROBOTWIN" env.train.video_cfg.save_video=false
  "env.train.task_config.save_path=$B_RUN/robotwin_data/train"
  env.train.task_config.task_name=move_stapler_pad env.train.task_config.step_lim=400
  env.eval.total_num_envs=8 env.eval.rollout_epoch=1
  env.eval.max_episode_steps=400 env.eval.max_steps_per_rollout_epoch=400
  env.eval.use_fixed_reset_state_ids=true "env.eval.assets_path=$ROBOTWIN" env.eval.video_cfg.save_video=false
  "env.eval.video_cfg.video_base_dir=$B_RUN/video/eval"
  "env.eval.task_config.save_path=$B_RUN/robotwin_data/eval"
  env.eval.task_config.task_name=move_stapler_pad env.eval.task_config.step_lim=400
  actor.global_batch_size=512 actor.micro_batch_size=32
  "actor.model.model_path=$MODEL" actor.model.num_steps=10 actor.model.openpi.num_steps=10
  ++actor.fsdp_config.checkpoint_format=local_shard
)

"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${A_ARGS[@]}" --cfg job --resolve > "$A_PACKET/resolved.yaml"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${B_ARGS[@]}" --cfg job --resolve > "$B_PACKET/resolved.yaml"

"$VENV/bin/python" - "$A_PACKET/resolved.yaml" "$B_PACKET/resolved.yaml" "$MODEL/conversion_manifest.json" "$A_PACKET/contract.json" "$B_PACKET/contract.json" <<'PY'
from __future__ import annotations

import json
import sys
import yaml

a = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
b = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
manifest = json.load(open(sys.argv[3], encoding="utf-8"))

assert manifest["source_revision"] == "e49e2ab6c11f07511573b67261bd129e88d0a416"
assert manifest["source_keys"] == manifest["target_keys"] == 813
assert not manifest["missing_keys"] and not manifest["unexpected_keys"]
assert not manifest["shape_mismatches"]
assert manifest["dtype_mismatch_count"] >= 0
assert "preserved" in manifest["dtype_note"]

for cfg in (a, b):
    assert cfg["actor"]["model"]["model_path"].endswith("sidney-pi05-robotwin-e49e2ab")
    assert cfg["actor"]["model"]["num_action_chunks"] == 50
    assert cfg["actor"]["model"]["action_dim"] == 14
    assert cfg["actor"]["model"]["num_steps"] == 10
    assert cfg["actor"]["model"]["openpi"]["config_name"] == "pi05_sidney_robotwin"
    assert cfg["actor"]["model"]["openpi"]["action_chunk"] == 50
    assert cfg["actor"]["model"]["openpi"]["num_steps"] == 10
    assert cfg["actor"]["model"]["openpi"]["num_images_in_input"] == 3
    assert cfg["actor"]["model"]["openpi"]["noise_level"] == 0.3
    assert cfg["actor"]["model"]["openpi"]["train_expert_only"] is True
    assert cfg["actor"]["optim"]["lr"] == 5e-6
    assert cfg["actor"]["fsdp_config"]["checkpoint_format"] == "local_shard"
    assert cfg["env"]["eval"]["max_episode_steps"] == 400
    assert cfg["env"]["eval"]["max_steps_per_rollout_epoch"] == 400
    assert cfg["env"]["eval"]["task_config"]["step_lim"] == 400
    assert cfg["env"]["train"]["center_crop"] is False
    assert cfg["env"]["eval"]["center_crop"] is False
    for split in ("train", "eval"):
        task = cfg["env"][split]["task_config"]
        assert task["embodiment"] == ["aloha-agilex"]
        assert task["camera"]["collect_wrist_camera"] is True
        assert task["domain_randomization"] == {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1,
            "random_head_camera_dis": 0,
            "random_table_height": 0,
            "random_light": False,
            "crazy_random_light_rate": 0,
        }

assert a["cluster"]["component_placement"] == {"actor, env, rollout": "4"}
assert a["runner"]["only_eval"] is True
assert a["env"]["eval"]["total_num_envs"] == 1
assert a["env"]["eval"]["rollout_epoch"] == 1
assert a["env"]["eval"]["task_config"]["task_name"] == "adjust_bottle"
assert a["env"]["eval"]["use_fixed_reset_state_ids"] is True
assert a["env"]["eval"]["auto_reset"] is True
assert a["env"]["eval"]["ignore_terminations"] is True
assert a["env"]["eval"]["is_eval"] is True
assert a["env"]["eval"]["seeds_path"].endswith("eval-seeds-adjust-bad1001.json")

assert b["cluster"]["component_placement"] == {"actor, env, rollout": "4,5"}
assert b["runner"]["only_eval"] is False
assert b["runner"]["max_steps"] == 1
assert b["runner"]["val_check_interval"] == 1 and b["runner"]["save_interval"] == 1
assert b["algorithm"]["group_size"] == 8
assert b["algorithm"]["update_epoch"] == 2
assert b["algorithm"]["logprob_type"] == "chunk_level"
assert b["algorithm"]["dvac_gradient_weighting"]["mode"] == "off"
assert b["env"]["train"]["total_num_envs"] == 64
assert b["env"]["train"]["rollout_epoch"] == 1
assert b["env"]["train"]["max_episode_steps"] == 400
assert b["env"]["train"]["task_config"]["task_name"] == "move_stapler_pad"
assert b["env"]["train"]["task_config"]["step_lim"] == 400
assert b["env"]["eval"]["total_num_envs"] == 8
assert b["env"]["eval"]["auto_reset"] is True
assert b["env"]["eval"]["ignore_terminations"] is True
assert b["env"]["eval"]["use_fixed_reset_state_ids"] is True
assert b["env"]["eval"]["is_eval"] is True
assert b["actor"]["global_batch_size"] == 512
assert b["actor"]["micro_batch_size"] == 32

a_summary = {
    "kind": "RLinf B=1 fixed inference plus known-UnStable retry",
    "physical_gpus": [4], "task": "adjust_bottle", "requested_seed": 1001,
    "episodes": 1, "physics_action_limit": 400, "max_policy_queries": 8,
    "H": 50, "C": 50, "M": 10, "optimizer_steps": 0,
}
b_summary = {
    "kind": "RLinf Sidney pi0.5 GRPO one-outer-step smoke",
    "physical_gpus": [4, 5], "task": "move_stapler_pad",
    "train_envs": 64, "rollout_epochs": 1, "trajectories": 64,
    "group_size": 8, "groups": 8, "physics_action_limit": 400,
    "H": 50, "C": 50, "M": 10, "max_query_records": 512,
    "global_batch": 512, "micro_batch": 32, "update_epochs": 2,
    "max_optimizer_steps": 2, "max_record_presentations": 1024,
    "fixed_eval_episodes": 8, "save_step": 1,
}
for path, payload in ((sys.argv[4], a_summary), (sys.argv[5], b_summary)):
    with open(path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)
        out.write("\n")
print(json.dumps({"A": a_summary, "B": b_summary}, indent=2))
PY

printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${A_ARGS[@]}" > "$A_PACKET/command.txt"
printf '\n' >> "$A_PACKET/command.txt"
printf '%q ' "$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" "${B_ARGS[@]}" > "$B_PACKET/command.txt"
printf '\n' >> "$B_PACKET/command.txt"
printf '%s\n' "$HEAD" > "$A_PACKET/source-head.txt"
printf '%s\n' "$HEAD" > "$B_PACKET/source-head.txt"
printf '%s\n' "$HEAD" > "$P_PACKET/source-head.txt"
sha256sum "$MODEL/conversion_manifest.json" "$MODEL/model.safetensors" "$MODEL/physical-intelligence/robotwin/norm_stats.json" > "$A_PACKET/model-sha256.txt"
cp "$A_PACKET/model-sha256.txt" "$B_PACKET/model-sha256.txt"
cp "$A_PACKET/model-sha256.txt" "$P_PACKET/model-sha256.txt"
sha256sum "$NATIVE_MODEL/model.safetensors" >> "$P_PACKET/model-sha256.txt"
printf '%s\n' prepared > "$P_PACKET/packet-complete.txt"
printf '%s\n' prepared > "$A_PACKET/packet-complete.txt"
printf '%s\n' prepared > "$B_PACKET/packet-complete.txt"

printf 'P_PACKET=%s\nA_PACKET=%s\nB_PACKET=%s\nSIDNEY_TWO_LEVEL_SMOKE_PACKET_PREPARED\n' "$P_PACKET" "$A_PACKET" "$B_PACKET"
