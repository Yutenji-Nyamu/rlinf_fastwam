#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
BOOTSTRAP=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821
PACKET="$BOOTSTRAP/resolved-packet-4gpu-official-half-v1"
SFT_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed64-4gpu4567-official-half-v1
PPO_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1
RELOAD_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-reload-fixed64-4gpu4567-v1
CKPT="$PPO_RUN/robotwin_ppo_openpi/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 0008ae6800df9f75fc8de7098bacb01735fd8fd2
test ! -e "$PACKET"
test ! -e "$SFT_RUN"
test ! -e "$PPO_RUN"
test ! -e "$RELOAD_RUN"
test -s "$MODEL/physical-intelligence/robotwin/norm_stats.json"

source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES
export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROOT/examples/embodiment"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$ROOT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
mkdir -p "$PACKET"

EVAL_PLACEMENT='cluster.component_placement={env\,\ rollout:4-7}'
PPO_PLACEMENT='cluster.component_placement={actor\,\ env\,\ rollout:4-7}'

COMMON_EVAL=(
  --config-path "$ROOT/evaluations/robotwin"
  --config-name robotwin_adjust_bottle_openpi_eval
  "$EVAL_PLACEMENT"
  "rollout.model.model_path=$MODEL"
  "env.eval.assets_path=$ROBOTWIN"
  "env.eval.total_num_envs=64"
  "env.eval.rollout_epoch=1"
  "env.eval.max_episode_steps=200"
  "env.eval.max_steps_per_rollout_epoch=200"
  "env.eval.use_fixed_reset_state_ids=true"
)

"$VENV/bin/python" "$ROOT/evaluations/eval_embodied_agent.py" \
  "${COMMON_EVAL[@]}" "runner.logger.log_path=$SFT_RUN" \
  --cfg job --resolve > "$PACKET/sft-fixed64.resolved.yaml"

"$VENV/bin/python" "$ROOT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_ppo_openpi \
  "$PPO_PLACEMENT" \
  "runner.logger.log_path=$PPO_RUN" \
  "runner.max_epochs=1" \
  "runner.max_steps=1" \
  "runner.val_check_interval=1" \
  "runner.save_interval=1" \
  "algorithm.update_epoch=1" \
  "env.train.total_num_envs=128" \
  "env.train.rollout_epoch=1" \
  "env.train.max_steps_per_rollout_epoch=50" \
  "env.train.assets_path=$ROBOTWIN" \
  "env.eval.total_num_envs=64" \
  "env.eval.rollout_epoch=1" \
  "env.eval.max_episode_steps=200" \
  "env.eval.max_steps_per_rollout_epoch=200" \
  "env.eval.use_fixed_reset_state_ids=true" \
  "env.eval.assets_path=$ROBOTWIN" \
  "actor.micro_batch_size=32" \
  "actor.global_batch_size=128" \
  "actor.model.model_path=$MODEL" \
  --cfg job --resolve > "$PACKET/ppo-oneopt.resolved.yaml"

"$VENV/bin/python" "$ROOT/evaluations/eval_embodied_agent.py" \
  "${COMMON_EVAL[@]}" "runner.logger.log_path=$RELOAD_RUN" \
  "runner.ckpt_path=$CKPT" \
  --cfg job --resolve > "$PACKET/ppo-reload-fixed64.resolved.yaml"

"$VENV/bin/python" - "$PACKET" "$ROOT" "$ROBOTWIN" "$MODEL" "$CKPT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import torch
from rlinf.envs.robotwin.seed_utils import partition_success_seeds

packet = Path(sys.argv[1])
root, robotwin, model, ckpt = sys.argv[2:]
sft = OmegaConf.load(packet / "sft-fixed64.resolved.yaml")
ppo = OmegaConf.load(packet / "ppo-oneopt.resolved.yaml")
reload = OmegaConf.load(packet / "ppo-reload-fixed64.resolved.yaml")

for cfg in (sft, reload):
    assert OmegaConf.to_container(cfg.cluster.component_placement) == {"env, rollout": "4-7"}
    assert cfg.env.eval.total_num_envs == 64
    assert cfg.env.eval.rollout_epoch == 1
    assert cfg.env.eval.max_episode_steps == 200
    assert cfg.env.eval.max_steps_per_rollout_epoch == 200
    assert cfg.env.eval.use_fixed_reset_state_ids is True
    assert cfg.env.eval.assets_path == robotwin
    assert cfg.rollout.model.model_path == model
    assert cfg.rollout.model.num_action_chunks == 50
    assert cfg.rollout.model.action_dim == 14
    assert cfg.rollout.model.openpi.config_name == "pi0_aloha_robotwin"
    assert cfg.rollout.model.openpi.num_images_in_input == 3
assert reload.runner.ckpt_path == ckpt

assert OmegaConf.to_container(ppo.cluster.component_placement) == {"actor, env, rollout": "4-7"}
assert ppo.runner.max_epochs == 1 and ppo.runner.max_steps == 1
assert ppo.runner.val_check_interval == 1 and ppo.runner.save_interval == 1
assert ppo.algorithm.update_epoch == 1
assert ppo.env.train.total_num_envs == 128
assert ppo.env.train.rollout_epoch == 1
assert ppo.env.train.max_steps_per_rollout_epoch == 50
assert ppo.env.train.max_episode_steps == 200
assert ppo.env.train.assets_path == robotwin
assert ppo.env.eval.total_num_envs == 64
assert ppo.env.eval.assets_path == robotwin
assert ppo.actor.micro_batch_size == 32 and ppo.actor.global_batch_size == 128
assert ppo.actor.model.model_path == model
assert ppo.actor.model.num_action_chunks == 50
assert ppo.actor.model.action_dim == 14

records = (
    ppo.env.train.total_num_envs
    * ppo.env.train.rollout_epoch
    * ppo.env.train.max_steps_per_rollout_epoch
    // ppo.actor.model.num_action_chunks
)
optimizer_steps = records // ppo.actor.global_batch_size * ppo.algorithm.update_epoch
gradient_accumulation = ppo.actor.global_batch_size // (ppo.actor.micro_batch_size * 4)
assert records == 128 and optimizer_steps == 1 and gradient_accumulation == 1

seed_data = json.loads(Path(root, "rlinf/envs/robotwin/seeds/eval_seeds.json").read_text())
success_seeds = torch.tensor(seed_data["adjust_bottle"]["success_seeds"], dtype=torch.long)
fixed64 = []
for rank in range(4):
    worker = partition_success_seeds(
        success_seeds,
        base_seed=int(sft.env.eval.seed),
        seed_offset=rank,
        total_num_processes=4,
        num_group=16,
    )
    assert worker.numel() == 32
    fixed64.extend(int(x) for x in worker[:16])
assert len(fixed64) == len(set(fixed64)) == 64

summary = {
    "driver_cuda_visible_devices": None,
    "placement_physical_device_indices": [4, 5, 6, 7],
    "base_seed": int(sft.env.eval.seed),
    "success_seed_count": int(success_seeds.numel()),
    "env_world_size": 4,
    "train_total_envs": 128,
    "train_envs_per_worker": 32,
    "eval_total_envs": 64,
    "eval_envs_per_worker": 16,
    "fixed64_initial_reset_ids": fixed64,
    "train_chunk_records": int(records),
    "actor_micro_batch_size": int(ppo.actor.micro_batch_size),
    "actor_global_batch_size": int(ppo.actor.global_batch_size),
    "gradient_accumulation_steps": int(gradient_accumulation),
    "distributed_optimizer_steps": int(optimizer_steps),
    "checkpoint_full_weights": ckpt,
}
(packet / "budget-and-seeds.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

sha256sum "$PACKET"/*
printf '%s\n' 'R2_FOUR_GPU_OFFICIAL_HALF_PACKET_COMPOSE_OK'
