#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/robotwin-native/RoboTwin/data/sz_collect_smoke_1ep_20260821
CFG=/data/chenyiteng/projects/robotwin-native/RoboTwin/env_cfg/task_config/sz_collect_smoke_1ep_20260821.yml
EXPECTED_ROOT=/data/chenyiteng/projects/robotwin-native/RoboTwin/data/sz_collect_smoke_1ep_20260821
EXPECTED_CFG=/data/chenyiteng/projects/robotwin-native/RoboTwin/env_cfg/task_config/sz_collect_smoke_1ep_20260821.yml

test "$(realpath "$ROOT")" = "$EXPECTED_ROOT"
test "$(realpath "$CFG")" = "$EXPECTED_CFG"
test -s "$ROOT/adjust_bottle/aloha_agilex/data/episode_0000000.hdf5"
test -s "$ROOT/adjust_bottle/aloha_agilex/video/episode_0000000.mp4"
test -s "$ROOT/adjust_bottle/aloha_agilex/instruction/episode_0000000.json"

rm -rf -- "$ROOT"
rm -f -- "$CFG"
test ! -e "$ROOT"
test ! -e "$CFG"
printf 'removed_exact_smoke_root=%s\nremoved_exact_smoke_config=%s\n' "$ROOT" "$CFG"
