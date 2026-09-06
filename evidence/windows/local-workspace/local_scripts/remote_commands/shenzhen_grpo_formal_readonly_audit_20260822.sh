#!/usr/bin/env bash
set -euo pipefail

GRPO_REPO=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
GRPO_YAML="$GRPO_REPO/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml"
PPO_YAML="$GRPO_REPO/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml"
PACKET=/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/grpo-smoke1-current-4gpu32x8-g8-v1
GRPO_RESOLVED="$PACKET/smoke.resolved.yaml"
PPO_RESOLVED=/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1/resolved.yaml
CANDIDATE_RUN=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-smoke1-current-4gpu32x8-g8-v1
BASE=7d07a4212ee6858cc333e1d4fab7a37256d1f839
OLD=6d0db56bf26f972cd27fa29535f5eb939e80e5bf

echo '=== IDENTITY ==='
date --iso-8601=seconds
id
hostname

echo '=== CURRENT GRPO GIT ==='
git -C "$GRPO_REPO" rev-parse HEAD HEAD^ "$BASE"
git -C "$GRPO_REPO" branch --show-current
git -C "$GRPO_REPO" status --short
git -C "$GRPO_REPO" diff --name-status "$BASE"..HEAD
git -C "$GRPO_REPO" diff --numstat "$BASE"..HEAD
git -C "$GRPO_REPO" rev-parse '@{upstream}'
git -C "$GRPO_REPO" rev-parse personal/codex/sz-7d07a421-grpo-pi0-robotwin
sha256sum "$GRPO_YAML" "$PPO_YAML" "$GRPO_RESOLVED" "$PPO_RESOLVED" "$PACKET/budget.json"
if [ -e "$CANDIDATE_RUN" ]; then
  echo 'candidate_run_exists=true'
else
  echo 'candidate_run_exists=false'
fi

echo '=== OLD/CURRENT CORE SOURCE AUDIT ==='
OLD="$OLD" BASE="$BASE" GRPO_REPO="$GRPO_REPO" \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import ast
import hashlib
import json
import os
import subprocess

repo = os.environ["GRPO_REPO"]
old = os.environ["OLD"]
base = os.environ["BASE"]


def git_show(rev, path):
    return subprocess.check_output(
        ["git", "-C", repo, "show", f"{rev}:{path}"], text=True
    )


def function_ast(rev, path, name):
    tree = ast.parse(git_show(rev, path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise RuntimeError(f"missing {name} in {rev}:{path}")


checks = []
for path, name in [
    ("rlinf/algorithms/advantages.py", "compute_grpo_advantages"),
    ("rlinf/algorithms/losses.py", "compute_grpo_actor_loss_fn"),
]:
    old_ast = function_ast(old, path, name)
    new_ast = function_ast(base, path, name)
    checks.append(
        {
            "path": path,
            "symbol": name,
            "semantic_ast_equal": old_ast == new_ast,
            "old_ast_sha256": hashlib.sha256(old_ast.encode()).hexdigest(),
            "new_ast_sha256": hashlib.sha256(new_ast.encode()).hexdigest(),
        }
    )

for path in [
    "examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml",
    "examples/embodiment/config/model/pi0.yaml",
    "rlinf/envs/robotwin/robotwin_env.py",
]:
    old_blob = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", f"{old}:{path}"], text=True
    ).strip()
    new_blob = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", f"{base}:{path}"], text=True
    ).strip()
    checks.append(
        {
            "path": path,
            "blob_equal": old_blob == new_blob,
            "old_blob": old_blob,
            "new_blob": new_blob,
        }
    )

worker = git_show(base, "rlinf/workers/actor/embodied_fsdp_actor_worker.py")
for symbol in [
    "compute_grpo_advantages",
    "compute_grpo_actor_loss_fn",
    "filter_rewards",
    "group_size",
]:
    checks.append(
        {
            "path": "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
            "symbol": symbol,
            "present": symbol in worker,
        }
    )
print(json.dumps(checks, indent=2, sort_keys=True))
PY

echo '=== SELECTED RESOLVED CONFIGS ==='
GRPO_RESOLVED="$GRPO_RESOLVED" PPO_RESOLVED="$PPO_RESOLVED" \
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import json
import os
from pathlib import Path

import yaml


def get(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


paths = [
    "cluster.component_placement",
    "runner.max_epochs",
    "runner.max_steps",
    "runner.val_check_interval",
    "runner.save_interval",
    "runner.only_eval",
    "algorithm.adv_type",
    "algorithm.loss_type",
    "algorithm.normalize_advantages",
    "algorithm.group_size",
    "algorithm.reward_type",
    "algorithm.logprob_type",
    "algorithm.update_epoch",
    "algorithm.loss_agg_func",
    "algorithm.kl_beta",
    "algorithm.clip_ratio_high",
    "algorithm.clip_ratio_low",
    "algorithm.filter_rewards",
    "algorithm.rewards_lower_bound",
    "algorithm.rewards_upper_bound",
    "env.enable_offload",
    "env.train.enable_offload",
    "env.train.total_num_envs",
    "env.train.rollout_epoch",
    "env.train.group_size",
    "env.train.max_episode_steps",
    "env.train.max_steps_per_rollout_epoch",
    "env.train.use_fixed_reset_state_ids",
    "env.train.video_cfg.save_video",
    "env.eval.enable_offload",
    "env.eval.total_num_envs",
    "env.eval.rollout_epoch",
    "env.eval.group_size",
    "env.eval.use_fixed_reset_state_ids",
    "env.eval.video_cfg.save_video",
    "actor.enable_offload",
    "actor.micro_batch_size",
    "actor.global_batch_size",
    "actor.model.add_value_head",
    "actor.model.num_action_chunks",
    "actor.model.action_dim",
    "actor.model.num_steps",
    "actor.model.openpi.config_name",
    "actor.model.openpi.action_chunk",
    "actor.model.openpi.num_steps",
    "actor.model.openpi.noise_method",
    "actor.optim.lr",
    "actor.optim.clip_grad",
    "actor.fsdp_config.gradient_checkpointing",
    "actor.fsdp_config.cpu_offload",
    "rollout.enable_offload",
    "rollout.recompute_logprobs",
    "critic.use_critic_model",
    "reward.use_reward_model",
]

result = {}
for label, path in [
    ("grpo_smoke_resolved", os.environ["GRPO_RESOLVED"]),
    ("ppo_formal_resolved", os.environ["PPO_RESOLVED"]),
]:
    data = yaml.safe_load(Path(path).read_text())
    result[label] = {key: get(data, key) for key in paths}
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo '=== CURRENT WORKER GRPO DISPATCH SITES ==='
grep -n -E 'filter_rewards|adv_type|loss_type|advantage|actor_loss|group_size' \
  "$GRPO_REPO/rlinf/workers/actor/embodied_fsdp_actor_worker.py" | \
  sed -n '1,160p'

echo '=== PACKET BUDGET ==='
sed -n '1,240p' "$PACKET/budget.json"
echo 'SZ_GRPO_FORMAL_READONLY_AUDIT_OK'
