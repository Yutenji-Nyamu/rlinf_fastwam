#!/usr/bin/env bash
set -euo pipefail

REPO=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-dvac-grpo-current
OLD=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v2/resolved.yaml
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
echo '=== source-locked official 7d07 GRPO YAML selected lines ==='
configs=(
  examples/embodiment/config/libero_spatial_grpo_openpi.yaml
  examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla.yaml
  examples/embodiment/config/robotwin_beat_block_hammer_grpo_openvlaoft.yaml
)
printf 'exact_robotwin_pi0_grpo_config='; git -C "$REPO" ls-tree -r --name-only 7d07a421 | grep -q '^examples/embodiment/config/robotwin_.*grpo_openpi.yaml$' && echo present || echo absent
for cfg in "${configs[@]}"; do
  echo "--- $cfg ---"
  git -C "$REPO" show "7d07a421:$cfg" | grep -nE 'total_num_envs|rollout_epoch|global_batch_size|micro_batch_size|update_epoch|group_size|max_steps_per_rollout_epoch|num_action_chunks' || true
done

echo '=== Shenzhen config-only commit 554c6dc8 pi0 RoboTwin GRPO defaults ==='
git -C "$REPO" show 554c6dc8d586162d9444c01fa88308ed4f5203d0:examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml \
  | grep -nE 'total_num_envs|rollout_epoch|global_batch_size|micro_batch_size|update_epoch|group_size|max_steps_per_rollout_epoch|num_action_chunks' || true

echo '=== Shenzhen GRPO v2 resolved budget ==='
"$PY" - "$OLD" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
trajectories = cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch
records = trajectories * (cfg.env.train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks)
print(f"train_envs={cfg.env.train.total_num_envs}")
print(f"rollout_epoch={cfg.env.train.rollout_epoch}")
print(f"trajectories={trajectories}")
print(f"groups={trajectories // cfg.algorithm.group_size}")
print(f"max_chunk_records={records}")
print(f"global_batch={cfg.actor.global_batch_size}")
print(f"micro_batch={cfg.actor.micro_batch_size}")
print(f"update_epoch={cfg.algorithm.update_epoch}")
print(f"video_train={cfg.env.train.video_cfg.save_video}")
print(f"video_eval={cfg.env.eval.video_cfg.save_video}")
PY
