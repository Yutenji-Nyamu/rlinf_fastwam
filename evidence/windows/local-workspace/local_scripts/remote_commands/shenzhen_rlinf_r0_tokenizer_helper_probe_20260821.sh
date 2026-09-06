#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test -z "$(git -C "$ROOT" status --porcelain)"

printf '%s\n' '=== download_assets.sh ==='
nl -ba "$ROOT/requirements/embodied/download_assets.sh" | sed -n '1,220p'
printf '%s\n' '=== openpi asset references ==='
rg -n 'openpi_tokenizer|paligemma_tokenizer|download_assets|install_uv|install_robotwin_env' \
  "$ROOT/requirements/install.sh" \
  "$ROOT/requirements/embodied" || true
printf '%s\n' '=== current tokenizer cache ==='
if test -e /home/chenyiteng/.cache/openpi; then
  find /home/chenyiteng/.cache/openpi -maxdepth 3 -printf '%y %s %p\n' | sort
else
  printf '%s\n' 'ABSENT /home/chenyiteng/.cache/openpi'
fi
