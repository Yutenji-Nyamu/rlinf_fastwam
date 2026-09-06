#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/RLinf_ogpo_pi0_robotwin
base=6d0db56bf26f972cd27fa29535f5eb939e80e5bf
qam=ff8e28ef6a4d485642e695b6b76c5c84187e134e
rlt=2b8199d8ab2e7b110994fd3234bf7007196c3af9

echo '=== target-status ==='
git -C "$repo" status --short --branch
git -C "$repo" rev-parse HEAD

echo '=== relevant-tree ==='
find "$repo/rlinf/workers/actor" -maxdepth 2 -type f -name '*.py' -printf '%P\n' | sort
find "$repo/rlinf/models/embodiment" -maxdepth 3 -type f -name '*.py' -printf '%P\n' | sort
find "$repo/rlinf/algorithms" -maxdepth 2 -type f -name '*.py' -printf '%P\n' | sort
find "$repo/rlinf/data" -maxdepth 2 -type f -name '*.py' -printf '%P\n' | sort 2>/dev/null || true

echo '=== dispatch-symbols ==='
{ git -C "$repo" grep -n -E 'loss_type|EmbodiedFSDPActor|FSDP.*Policy|actor_cls|actor_class' -- \
  examples/embodiment/train_embodied_agent.py rlinf/config.py rlinf/workers/actor || true; } | sed -n '1,240p'

echo '=== lifecycle-symbols ==='
{ git -C "$repo" grep -n -E 'class .*Actor|recv_rollout_trajectories|compute_advantages_and_returns|run_training|sync_model_to_rollout|save_checkpoint|load_checkpoint' -- \
  rlinf/workers/actor rlinf/runners/embodied_runner.py || true; } | sed -n '1,320p'

echo '=== openpi-symbols ==='
{ git -C "$repo" grep -n -E 'class OpenPi|predict_action_batch|_build_prefix_cache|sample_mean_var|flow_sde|joint_logprob|default_forward|train_expert_only|freeze_vlm' -- \
  rlinf/models/embodiment/openpi || true; } | sed -n '1,320p'

echo '=== env-trajectory-symbols ==='
{ git -C "$repo" grep -n -E 'chunk_step|env_interact_step|to_trajectory|forward_inputs|bootstrap|auto_reset|final_observation' -- \
  rlinf/workers/env/env_worker.py rlinf/envs/robotwin/robotwin_env.py rlinf/data rlinf/schemas 2>/dev/null || true; } | sed -n '1,320p'

echo '=== qam-narrow-diff ==='
git -C "$repo" diff --name-status "$base..$qam" -- \
  examples/embodiment/train_embodied_agent.py rlinf/config.py rlinf/workers/actor \
  rlinf/models/embodiment rlinf/algorithms rlinf/data tests | sed -n '1,240p'

echo '=== rlt-narrow-diff ==='
git -C "$repo" diff --name-status "$base..$rlt" -- \
  examples/embodiment/train_embodied_agent.py rlinf/config.py rlinf/workers/actor \
  rlinf/models/embodiment rlinf/algorithms rlinf/data tests | sed -n '1,240p'
