#!/usr/bin/env bash
set -euo pipefail

COMMIT=554c6dc8d586162d9444c01fa88308ed4f5203d0
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
MODEL=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/grpo-smoke1-current-4gpu32x8-g8-v1
RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-smoke1-current-4gpu32x8-g8-v1

printf 'MARKER=SZ_GRPO_SMOKE_PACKET_COMPOSE_V1\n'
TZ=Asia/Shanghai date --iso-8601=seconds
id

test "$(git -C "$WT" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$WT" status --short)"
test "$(git -C "$WT" rev-parse personal/codex/sz-7d07a421-grpo-pi0-robotwin)" = "$COMMIT"
test ! -e "$PACKET"
test ! -e "$RUN"
test -d "$ROBOTWIN"
test -s "$MODEL/physical-intelligence/robotwin/norm_stats.json"

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$WT:$ROBOTWIN${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

install -d -m 755 "$PACKET"
PLACEMENT='cluster.component_placement={actor\,\ env\,\ rollout:4-7}'
"$VENV/bin/python" "$WT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$WT/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  "$PLACEMENT" \
  "runner.logger.log_path=$RUN" \
  "runner.max_epochs=1" \
  "runner.max_steps=1" \
  "runner.val_check_interval=1" \
  "runner.save_interval=1" \
  "env.train.assets_path=$ROBOTWIN" \
  "env.eval.assets_path=$ROBOTWIN" \
  "actor.model.model_path=$MODEL" \
  --cfg job --resolve > "$PACKET/smoke.resolved.yaml"

"$VENV/bin/python" - "$PACKET/smoke.resolved.yaml" "$ROBOTWIN" "$MODEL" "$RUN" "$PACKET" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

resolved_path = Path(sys.argv[1])
robotwin, model, run, packet = sys.argv[2:]
cfg = OmegaConf.load(resolved_path)

assert OmegaConf.to_container(cfg.cluster.component_placement) == {
    "actor, env, rollout": "4-7"
}
assert cfg.runner.logger.log_path == run
assert cfg.runner.max_epochs == 1
assert cfg.runner.max_steps == 1
assert cfg.runner.val_check_interval == 1
assert cfg.runner.save_interval == 1
assert cfg.algorithm.adv_type == "grpo"
assert cfg.algorithm.loss_type == "actor"
assert cfg.algorithm.group_size == 8
assert cfg.algorithm.filter_rewards is True
assert cfg.algorithm.update_epoch == 2
assert cfg.env.train.total_num_envs == 32
assert cfg.env.train.rollout_epoch == 8
assert cfg.env.train.max_episode_steps == 200
assert cfg.env.train.max_steps_per_rollout_epoch == 200
assert cfg.env.eval.total_num_envs == 64
assert cfg.env.eval.rollout_epoch == 1
assert cfg.env.eval.use_fixed_reset_state_ids is True
assert cfg.env.train.assets_path == cfg.env.eval.assets_path == robotwin
assert cfg.actor.model.model_path == model
assert cfg.actor.model.add_value_head is False
assert cfg.actor.micro_batch_size == 32
assert cfg.actor.global_batch_size == 512

trajectories = cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch
groups = trajectories // cfg.algorithm.group_size
max_chunk_records = trajectories * (
    cfg.env.train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks
)
max_optimizer_steps = (
    max_chunk_records // cfg.actor.global_batch_size * cfg.algorithm.update_epoch
)
summary = {
    "source_commit": "554c6dc8d586162d9444c01fa88308ed4f5203d0",
    "runtime_launched": False,
    "physical_gpu_indices": [4, 5, 6, 7],
    "outer_steps": 1,
    "train_trajectories": int(trajectories),
    "train_groups": int(groups),
    "max_action_slots": int(
        trajectories * cfg.env.train.max_steps_per_rollout_epoch
    ),
    "max_chunk_records_before_reward_filter": int(max_chunk_records),
    "max_distributed_optimizer_steps_after_filtering": int(max_optimizer_steps),
    "eval_episodes": int(cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch),
    "checkpoints": ["global_step_1"],
    "run_path": run,
    "note": "One outer-step smoke; reward filtering makes the realized record and optimizer-step counts data-dependent, bounded above by the listed maxima.",
}
Path(packet, "budget.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

sha256sum "$PACKET/smoke.resolved.yaml" "$PACKET/budget.json"
du -ah "$PACKET" | sort -h
test ! -e "$RUN"
printf 'MARKER=SZ_GRPO_SMOKE_PACKET_COMPOSE_OK\n'
