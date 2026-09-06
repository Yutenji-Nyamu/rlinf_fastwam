#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
RLINF=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
ROBOTWIN=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support

printf '%s\n' '=== LOCKED SOURCE CUROBO REFERENCES ==='
grep -RIn --exclude-dir=.git --include='*.sh' --include='*.txt' --include='*.md' --include='*.rst' \
  'NVlabs/curobo\|nvidia-curobo\|curobo.git' "$RLINF/requirements" "$RLINF/docs" "$ROBOTWIN" \
  | head -n 160 || true

printf '%s\n' '=== UPSTREAM VERSION TAGS ==='
git ls-remote --tags https://github.com/NVlabs/curobo.git \
  | grep -E 'refs/tags/v?0\.[0-9]+\.[0-9]+(\^\{\})?$' \
  | tail -n 80

printf '%s\n' '=== KNOWN LEGACY TAG OBJECTS ==='
for tag in v0.7.8 v0.7.9 0.7.8 0.7.9; do
  printf '%s ' "$tag"
  git ls-remote https://github.com/NVlabs/curobo.git "refs/tags/$tag" "refs/tags/$tag^{}" || true
done
