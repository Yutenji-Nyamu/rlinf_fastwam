#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
printf '%s\n' '=== SEED FILES ==='
find "$ROOT/rlinf/envs/robotwin" -maxdepth 4 -type f \( -iname '*seed*' -o -name '*.json' \) -print | sort
printf '%s\n' '=== SEED UTILITY SOURCE ==='
nl -ba "$ROOT/rlinf/envs/robotwin/seed_utils.py" | sed -n '1,260p'
