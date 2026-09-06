#!/usr/bin/env bash
set -euo pipefail

remote_script='/home/chenyiteng/.local/share/codex-server-scripts/shenzhen_grpo_launch_formal100_4gpu128x4_g8_ppo_matched_20260822.sh'
expected_sha='79f73831d642533cb7eefd841633e05d4a2763675c386a5f5fe1ffbcdf284786'

date --iso-8601=seconds
hostname
id
test -f "$remote_script"
test "$(stat -c %s "$remote_script")" = 5000
actual_sha="$(sha256sum "$remote_script" | awk '{print $1}')"
test "$actual_sha" = "$expected_sha"
bash -n "$remote_script"
stat -c 'script size=%s mode=%a owner=%U:%G path=%n' "$remote_script"
printf 'sha256=%s\n' "$actual_sha"
printf 'bash_n=PASS\n'
