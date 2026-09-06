#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839

printf '%s\n' '=== placement config and parser callsites ==='
grep -R -n --include='*.py' --include='*.yaml' \
  -E 'component_placement|env_world_size|total_num_envs.*env_world_size' \
  "$ROOT/rlinf/utils/placement.py" "$ROOT/rlinf/config.py" \
  "$ROOT/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml" \
  "$ROOT/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml" | head -n 160

printf '%s\n' '=== seed utility ==='
sed -n '1,260p' "$ROOT/rlinf/envs/robotwin/seed_utils.py"

printf '%s\n' '=== robotwin env seed/reset callsites ==='
grep -R -n --include='*.py' \
  -E 'partition_success_seeds|reset_state_ids|use_fixed_reset_state_ids|success_seeds' \
  "$ROOT/rlinf/envs/robotwin" | head -n 200

printf '%s\n' 'R2_FOUR_GPU_SOURCE_PROBE_OK'
