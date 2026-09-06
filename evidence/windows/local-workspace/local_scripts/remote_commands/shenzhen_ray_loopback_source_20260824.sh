#!/usr/bin/env bash
set -u
raypkg=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/ray
if command -v rg >/dev/null 2>&1; then
  rg -n --glob '*.py' '127\.0\.0\.1|node_ip_address.*loopback|loopback.*node_ip_address' \
    "$raypkg/_private" "$raypkg/scripts" 2>/dev/null | head -n 160 || true
else
  grep -RIn --include='*.py' -E '127\.0\.0\.1|node_ip_address.*loopback|loopback.*node_ip_address' \
    "$raypkg/_private" "$raypkg/scripts" 2>/dev/null | head -n 160 || true
fi
printf '%s\n' '--- services.py 760-815 ---'
sed -n '760,815p' "$raypkg/_private/services.py"
printf '%s\n' '--- node.py 1365-1425 ---'
sed -n '1365,1425p' "$raypkg/_private/node.py"
printf '%s\n' '--- scripts.py 580-625 ---'
sed -n '580,625p' "$raypkg/scripts/scripts.py"
printf '%s\n' '--- address auto resolution ---'
grep -RIn --include='*.py' -E 'def (get_ray_address_from_environment|find_bootstrap_address|canonicalize_bootstrap_address)|RAY_ADDRESS' \
  "$raypkg/_private" 2>/dev/null | head -n 120 || true
