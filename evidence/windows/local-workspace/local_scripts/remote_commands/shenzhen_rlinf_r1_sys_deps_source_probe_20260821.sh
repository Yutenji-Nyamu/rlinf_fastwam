#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
printf '%s\n' '=== sys_deps.sh ==='
nl -ba "$ROOT/requirements/sys_deps.sh" | sed -n '1,320p'
