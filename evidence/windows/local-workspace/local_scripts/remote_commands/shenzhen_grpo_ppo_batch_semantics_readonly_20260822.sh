#!/usr/bin/env bash
set -euo pipefail

GRPO_REPO=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
PPO_RUN=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1

echo '=== SOURCE ==='
git -C "$GRPO_REPO" rev-parse HEAD
git -C "$GRPO_REPO" status --short

echo '=== GRPO CONFIG ==='
sed -n '1,240p' "$GRPO_REPO/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml"

echo '=== PPO RESOLVED RELEVANT ==='
grep -n -E 'total_num_envs|rollout_epoch|max_steps_per_rollout_epoch|global_batch_size|micro_batch_size|update_epoch|group_size|filter_rewards|reward_filter|adv_type|loss_type|add_value_head|use_critic_model|val_check_interval|save_interval|max_steps:' "$PPO_RUN/resolved.yaml"

echo '=== WORKER SYMBOL LOCATIONS ==='
grep -n -E 'filter_rewards|filter_trajectories|group_size|global_batch_size|micro_batch_size|update_epoch|compute_grpo_advantages|compute_values|reshape.*group|mini_batch|micro_batch|num_global|train_actor|update_actor' "$GRPO_REPO/rlinf/workers/actor/embodied_fsdp_actor_worker.py"

echo '=== WORKER SOURCE 1-520 ==='
sed -n '1,520p' "$GRPO_REPO/rlinf/workers/actor/embodied_fsdp_actor_worker.py"

echo '=== BATCH HELPERS LOCATIONS ==='
grep -R -n -E 'def (split|chunk|mini_batch|micro_batch)|global_batch_size|gradient_accumulation_steps' "$GRPO_REPO/rlinf" | head -n 240

echo 'SZ_GRPO_PPO_BATCH_SEMANTICS_READONLY_OK'
