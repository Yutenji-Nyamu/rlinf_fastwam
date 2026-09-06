#!/usr/bin/env bash
set -euo pipefail

remote_script='/home/chenyiteng/.local/share/codex-server-scripts/shenzhen_grpo_launch_formal100_4gpu128x4_g8_ppo_matched_20260822.sh'
test "$(stat -c %s "$remote_script")" = 5000
test "$(sha256sum "$remote_script" | awk '{print $1}')" = 79f73831d642533cb7eefd841633e05d4a2763675c386a5f5fe1ffbcdf284786
bash -n "$remote_script"
bash "$remote_script"
