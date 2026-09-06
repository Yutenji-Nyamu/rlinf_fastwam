#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/RLinf

printf '=== identity ===\n'
date --iso-8601=seconds
id
hostname

printf '\n=== canonical ===\n'
test -d "$ROOT/.git"
git -C "$ROOT" rev-parse HEAD
git -C "$ROOT" status --short --branch
git -C "$ROOT" remote -v

printf '\n=== worktrees ===\n'
git -C "$ROOT" worktree list --porcelain

printf '\n=== each RLinf worktree ===\n'
while IFS= read -r repo; do
  test -n "$repo" || continue
  printf '\n--- %s ---\n' "$repo"
  git -C "$repo" branch --show-current || true
  git -C "$repo" rev-parse HEAD
  git -C "$repo" status --short --branch
done < <(git -C "$ROOT" worktree list --porcelain | sed -n 's/^worktree //p')

printf '\n=== current-base relevant source ===\n'
for path in \
  rlinf/models/embodiment/openpi/openpi_action_model.py \
  rlinf/models/embodiment/openpi_rlinf/openpi_action_model.py \
  rlinf/workers/actor/embodied_fsdp_actor_worker.py \
  rlinf/workers/actor/fsdp_sac_policy_worker.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py \
  rlinf/workers/actor/fsdp_rlt_td3_policy_worker.py \
  rlinf/data/storage/replay/buffer.py \
  rlinf/algorithms/rlt/route.py \
  examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml \
  examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml \
  examples/embodiment/config/libero_spatial_dsrl_openpi.yaml; do
  if test -f "$ROOT/$path"; then
    printf 'present lines=%s sha256=%s path=%s\n' \
      "$(wc -l < "$ROOT/$path")" \
      "$(sha256sum "$ROOT/$path" | awk '{print $1}')" \
      "$path"
  else
    printf 'absent path=%s\n' "$path"
  fi
done

printf '\n=== no current Shenzhen RLT or DSRL port branch assumption ===\n'
git -C "$ROOT" branch -a --list '*rlt*' '*dsrl*'
