#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839

sed -n '35,115p' "$ROOT/rlinf/envs/robotwin/robotwin_env.py"
sed -n '410,510p' "$ROOT/rlinf/envs/robotwin/robotwin_env.py"
printf '%s\n' 'R2_FOUR_GPU_SEED_CALLSITE_PROBE_OK'
