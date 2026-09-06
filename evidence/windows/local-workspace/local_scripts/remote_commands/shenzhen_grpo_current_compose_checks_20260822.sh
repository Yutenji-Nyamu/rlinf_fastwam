#!/usr/bin/env bash
set -euo pipefail

BASE=7d07a4212ee6858cc333e1d4fab7a37256d1f839
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
OUT=/data/chenyiteng/results/rlinf-shenzhen/grpo/pretest-current-7d07a421-20260822-v1
CFG=examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml

printf 'MARKER=SZ_GRPO_CURRENT_COMPOSE_CHECKS_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id

test "$(git -C "$WT" rev-parse HEAD)" = "$BASE"
test "$(git -C "$WT" branch --show-current)" = codex/sz-7d07a421-grpo-pi0-robotwin
test -f "$WT/$CFG"
test ! -e "$OUT"
test -d "$ROBOTWIN"
test -s "$MODEL/physical-intelligence/robotwin/norm_stats.json"

printf '%s\n' '=== PRE-COMPOSE DIFF ==='
git -C "$WT" diff --check
git -C "$WT" status --short --branch
git -C "$WT" diff --stat
git -C "$WT" diff --no-index -- \
  "$WT/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml" \
  "$WT/$CFG" || test "$?" = 1

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

install -d -m 755 "$OUT"
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  "runner.logger.log_path=$OUT/not-launched" \
  "env.train.assets_path=$ROBOTWIN" \
  "env.eval.assets_path=$ROBOTWIN" \
  "actor.model.model_path=$MODEL" \
  --cfg job --resolve > "$OUT/baseline.resolved.yaml"

"$VENV/bin/python" - "$OUT/baseline.resolved.yaml" "$ROBOTWIN" "$MODEL" "$OUT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import torch

from rlinf.algorithms.advantages import compute_grpo_advantages

resolved_path = Path(sys.argv[1])
robotwin, model, out = sys.argv[2:]
cfg = OmegaConf.load(resolved_path)

assert OmegaConf.to_container(cfg.cluster.component_placement) == {
    "actor, env, rollout": "0-3"
}
assert cfg.algorithm.adv_type == "grpo"
assert cfg.algorithm.loss_type == "actor"
assert cfg.algorithm.group_size == 8
assert cfg.algorithm.normalize_advantages is True
assert cfg.algorithm.reward_type == "chunk_level"
assert cfg.algorithm.logprob_type == "chunk_level"
assert cfg.algorithm.filter_rewards is True
assert cfg.algorithm.rewards_lower_bound == 0.1
assert cfg.algorithm.rewards_upper_bound == 0.9
assert cfg.algorithm.update_epoch == 2
assert cfg.actor.model.add_value_head is False
assert cfg.actor.model.openpi.detach_critic_input is False
assert cfg.critic.use_critic_model is False
assert cfg.env.train.total_num_envs == 32
assert cfg.env.train.rollout_epoch == 8
assert cfg.env.train.group_size == 8
assert cfg.env.train.total_num_envs % cfg.algorithm.group_size == 0
assert cfg.env.eval.total_num_envs == 64
assert cfg.env.eval.rollout_epoch == 1
assert cfg.env.eval.group_size == 1
assert cfg.env.eval.use_fixed_reset_state_ids is True
assert cfg.env.train.assets_path == cfg.env.eval.assets_path == robotwin
assert cfg.actor.model.model_path == model
assert cfg.actor.model.num_action_chunks == 50
assert cfg.actor.model.action_dim == 14
assert cfg.actor.model.openpi.config_name == "pi0_aloha_robotwin"
assert cfg.actor.model.openpi.num_images_in_input == 3
assert cfg.actor.micro_batch_size == 32
assert cfg.actor.global_batch_size == 512

world_size = 4
trajectories = cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch
groups = trajectories // cfg.algorithm.group_size
max_chunks_per_trajectory = (
    cfg.env.train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks
)
max_chunk_records = trajectories * max_chunks_per_trajectory
gradient_accumulation = cfg.actor.global_batch_size // (
    cfg.actor.micro_batch_size * world_size
)
max_distributed_optimizer_steps = (
    max_chunk_records // cfg.actor.global_batch_size * cfg.algorithm.update_epoch
)
assert trajectories == 256
assert groups == 32
assert max_chunks_per_trajectory == 4
assert max_chunk_records == 1024
assert gradient_accumulation == 4
assert max_distributed_optimizer_steps == 4

rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
loss_mask = torch.ones(1, rewards.numel())
advantages, returns = compute_grpo_advantages(
    rewards=rewards,
    loss_mask=loss_mask,
    group_size=8,
)
assert returns is None
assert advantages.shape == loss_mask.shape
assert torch.isfinite(advantages).all()
assert torch.allclose(advantages.view(-1, 8).mean(dim=-1), torch.zeros(2), atol=1e-6)

summary = {
    "base": "7d07a4212ee6858cc333e1d4fab7a37256d1f839",
    "config": "robotwin_adjust_bottle_grpo_openpi",
    "placement_logical_ranks": [0, 1, 2, 3],
    "train_envs": int(cfg.env.train.total_num_envs),
    "rollout_epochs": int(cfg.env.train.rollout_epoch),
    "group_size": int(cfg.algorithm.group_size),
    "trajectories_per_outer_step": int(trajectories),
    "groups_per_outer_step": int(groups),
    "max_chunk_records_per_outer_step": int(max_chunk_records),
    "actor_micro_batch_size": int(cfg.actor.micro_batch_size),
    "actor_global_batch_size": int(cfg.actor.global_batch_size),
    "update_epoch": int(cfg.algorithm.update_epoch),
    "gradient_accumulation_steps": int(gradient_accumulation),
    "max_distributed_optimizer_steps_per_outer_step": int(max_distributed_optimizer_steps),
    "eval_episodes": int(cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch),
    "grpo_math_probe": "finite_zero_group_mean",
    "runtime_launched": False,
}
Path(out, "contract-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

printf '%s\n' '=== ARTIFACTS ==='
sha256sum "$WT/$CFG" "$OUT/baseline.resolved.yaml" "$OUT/contract-summary.json"
du -ah "$OUT" | sort -h
printf 'MARKER=SZ_GRPO_CURRENT_COMPOSE_CHECKS_OK\n'
