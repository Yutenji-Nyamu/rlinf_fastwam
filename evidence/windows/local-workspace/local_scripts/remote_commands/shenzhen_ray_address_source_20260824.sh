#!/usr/bin/env bash
set -u
raypkg=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/ray
grep -RIn --include='*.py' -E '^def (get_ray_address_from_environment|find_bootstrap_address|canonicalize_bootstrap_address|canonicalize_bootstrap_address_or_die)' \
  "$raypkg/_private" 2>/dev/null || true
printf '%s\n' '--- services address section ---'
sed -n '390,455p' "$raypkg/_private/services.py"
sed -n '620,770p' "$raypkg/_private/services.py"
