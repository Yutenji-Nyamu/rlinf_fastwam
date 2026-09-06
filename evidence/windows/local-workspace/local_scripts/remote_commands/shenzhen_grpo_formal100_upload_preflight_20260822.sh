#!/usr/bin/env bash
set -euo pipefail

script_dir='/home/chenyiteng/.local/share/codex-server-scripts'
remote_script="$script_dir/shenzhen_grpo_launch_formal100_4gpu128x4_g8_ppo_matched_20260822.sh"

date --iso-8601=seconds
hostname
id
install -d -m 700 "$script_dir"
test ! -e "$remote_script"
stat -c 'dir mode=%a owner=%U:%G path=%n' "$script_dir"
printf 'remote_target=%s\n' "$remote_script"
printf 'upload_preflight=PASS\n'
