#!/usr/bin/env bash
set -u

repo=/root/autodl-tmp/RLinf
old_base=6d0db56bf26f972cd27fa29535f5eb939e80e5bf
new_base=48a775db09c16c455aeba7b0600c920e7c80d534

echo '[ancestry]'
git -C "$repo" merge-base "$old_base" "$new_base"
git -C "$repo" merge-base --is-ancestor "$old_base" "$new_base"
echo "old_is_ancestor_exit=$?"

echo '[commits]'
git -C "$repo" log --oneline --decorate "$old_base..$new_base"

echo '[overall_stat]'
git -C "$repo" diff --stat "$old_base..$new_base"

echo '[changed_names]'
git -C "$repo" diff --name-status "$old_base..$new_base"

echo '[rlt_overlap]'
git -C "$repo" diff --name-status "$old_base..$new_base" -- \
  examples/sft/config \
  examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml \
  examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml \
  rlinf/algorithms/rlt \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/models/embodiment/modules/rlt_token_transformer.py \
  rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  rlinf/workers/actor/fsdp_sac_policy_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/runners/embodied_runner.py

echo '[rlt_sources_at_new_base]'
for path in \
  examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml \
  examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml \
  rlinf/algorithms/rlt/route.py \
  rlinf/algorithms/rlt/rollout.py \
  rlinf/algorithms/rlt/transition.py \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
do
  if git -C "$repo" cat-file -e "$new_base:$path" 2>/dev/null; then
    echo "present $path"
  else
    echo "missing $path"
  fi
done
