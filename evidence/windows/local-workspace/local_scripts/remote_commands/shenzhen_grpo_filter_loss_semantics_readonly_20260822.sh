#!/usr/bin/env bash
set -euo pipefail

REPO=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421

echo '=== GRPO ADVANTAGE ==='
grep -n -A100 -B20 'def compute_grpo_advantages' "$REPO/rlinf/algorithms/advantages.py"

echo '=== GRPO ACTOR LOSS ==='
grep -n -A90 -B20 'def compute_grpo_actor_loss_fn' "$REPO/rlinf/algorithms/losses.py"

echo '=== PPO ACTOR LOSS / MASKED MEAN ==='
grep -n -A150 -B20 'def compute_ppo_actor_loss' "$REPO/rlinf/algorithms/losses.py" || true
grep -n -A140 -B20 'def masked_mean' "$REPO/rlinf/utils/utils.py"

echo '=== LOSS REGISTRY ==='
grep -R -n -E 'compute_grpo_actor_loss_fn|loss_type.*actor|register.*actor' "$REPO/rlinf/algorithms" | head -n 120

echo 'SZ_GRPO_FILTER_LOSS_SEMANTICS_READONLY_OK'
