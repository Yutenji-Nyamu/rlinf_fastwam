#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
BOOTSTRAP=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821
PACKET="$BOOTSTRAP/resolved-packet-v1"
SFT_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed8-v1
PPO_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-v1
RELOAD_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-reload-fixed8-v1
CKPT="$PPO_RUN/robotwin_ppo_openpi/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"

test ! -e "$PACKET"
test ! -e "$SFT_RUN"
test ! -e "$PPO_RUN"
test ! -e "$RELOAD_RUN"
test -s "$MODEL/physical-intelligence/robotwin/norm_stats.json"

source "$VENV/bin/activate"
export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROOT/examples/embodiment"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$ROOT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
mkdir -p "$PACKET"

COMMON_EVAL=(
  --config-path "$ROOT/evaluations/robotwin"
  --config-name robotwin_adjust_bottle_openpi_eval
  "rollout.model.model_path=$MODEL"
  "env.eval.assets_path=$ROBOTWIN"
  "env.eval.total_num_envs=8"
  "env.eval.rollout_epoch=1"
  "env.eval.max_episode_steps=200"
  "env.eval.max_steps_per_rollout_epoch=200"
  "env.eval.use_fixed_reset_state_ids=true"
)

"$VENV/bin/python" "$ROOT/evaluations/eval_embodied_agent.py" \
  "${COMMON_EVAL[@]}" \
  "runner.logger.log_path=$SFT_RUN" \
  --cfg job --resolve > "$PACKET/sft-fixed8.resolved.yaml"

"$VENV/bin/python" "$ROOT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$ROOT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_ppo_openpi \
  "runner.logger.log_path=$PPO_RUN" \
  "runner.max_epochs=1" \
  "runner.max_steps=1" \
  "runner.val_check_interval=1" \
  "runner.save_interval=1" \
  "algorithm.update_epoch=1" \
  "env.train.total_num_envs=8" \
  "env.train.rollout_epoch=1" \
  "env.train.max_steps_per_rollout_epoch=50" \
  "env.train.assets_path=$ROBOTWIN" \
  "env.eval.total_num_envs=8" \
  "env.eval.rollout_epoch=1" \
  "env.eval.max_episode_steps=200" \
  "env.eval.max_steps_per_rollout_epoch=200" \
  "env.eval.use_fixed_reset_state_ids=true" \
  "env.eval.assets_path=$ROBOTWIN" \
  "actor.micro_batch_size=1" \
  "actor.global_batch_size=8" \
  "actor.model.model_path=$MODEL" \
  --cfg job --resolve > "$PACKET/ppo-oneopt.resolved.yaml"

"$VENV/bin/python" "$ROOT/evaluations/eval_embodied_agent.py" \
  "${COMMON_EVAL[@]}" \
  "runner.logger.log_path=$RELOAD_RUN" \
  "runner.ckpt_path=$CKPT" \
  --cfg job --resolve > "$PACKET/ppo-reload-fixed8.resolved.yaml"

"$VENV/bin/python" - "$PACKET" "$ROOT" "$ROBOTWIN" "$MODEL" "$CKPT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import torch

packet = Path(sys.argv[1])
root, robotwin, model, ckpt = sys.argv[2:]
sft = OmegaConf.load(packet / "sft-fixed8.resolved.yaml")
ppo = OmegaConf.load(packet / "ppo-oneopt.resolved.yaml")
reload = OmegaConf.load(packet / "ppo-reload-fixed8.resolved.yaml")

for cfg in (sft, reload):
    assert OmegaConf.to_container(cfg.cluster.component_placement) == {"env, rollout": "0-7"}
    assert cfg.env.eval.total_num_envs == 8
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

assert OmegaConf.to_container(ppo.cluster.component_placement) == {"actor, env, rollout": "0-7"}
assert ppo.runner.max_epochs == 1 and ppo.runner.max_steps == 1
assert ppo.runner.val_check_interval == 1 and ppo.runner.save_interval == 1
assert ppo.algorithm.update_epoch == 1
assert ppo.env.train.total_num_envs == 8
assert ppo.env.train.rollout_epoch == 1
assert ppo.env.train.max_steps_per_rollout_epoch == 50
assert ppo.env.train.max_episode_steps == 200
assert ppo.env.train.assets_path == robotwin
assert ppo.env.eval.total_num_envs == 8
assert ppo.env.eval.assets_path == robotwin
assert ppo.actor.micro_batch_size == 1 and ppo.actor.global_batch_size == 8
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
assert records == 8 and optimizer_steps == 1

seed_data = json.loads(Path(root, "rlinf/envs/robotwin/seeds/eval_seeds.json").read_text())
success_seeds = torch.tensor(seed_data["adjust_bottle"]["success_seeds"], dtype=torch.long)
generator = torch.Generator().manual_seed(int(sft.env.eval.seed))
shuffled = success_seeds[torch.randperm(success_seeds.numel(), generator=generator)]
seeds_per_worker = shuffled.numel() // 8
fixed8 = [int(shuffled[offset * seeds_per_worker]) for offset in range(8)]

assert int(ppo.env.eval.seed) == int(sft.env.eval.seed) == int(reload.env.eval.seed)
summary = {
    "base_seed": int(sft.env.eval.seed),
    "success_seed_count": int(success_seeds.numel()),
    "total_env_processes": 8,
    "seeds_per_worker": int(seeds_per_worker),
    "fixed8_initial_reset_ids": fixed8,
    "train_chunk_records": int(records),
    "distributed_optimizer_steps": int(optimizer_steps),
    "checkpoint_full_weights": ckpt,
}
(packet / "budget-and-seeds.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

sha256sum "$PACKET"/*
printf '%s\n' 'R2_FIXED8_ONEOPT_PACKET_COMPOSE_OK'
