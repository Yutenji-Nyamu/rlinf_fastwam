#!/usr/bin/env bash
set -euo pipefail

WT=/data/chenyiteng/projects/fastwam-standalone/worktrees/fastwam-dvac-observe-7faa711
EXPECTED=fc652fb49cd32350eca15734b5c7124c0b8c2c02

test "$(id -un)" = chenyiteng
test -e "$WT/.git"
test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED"
test "$(git -C "$WT" branch --show-current)" = codex/sz-fastwam-dvac-observe
test -z "$(git -C "$WT" status --porcelain=v1 --untracked-files=all)"
test -x /home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python
test -f /data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt
test -f /data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json
test -d /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin/assets
test -d /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/third_party/RoboTwin/task_config

printf 'HEAD=%s\n' "$(git -C "$WT" rev-parse HEAD)"
printf 'BRANCH=%s\n' "$(git -C "$WT" branch --show-current)"
printf 'STATUS_LINES=%s\n' "$(git -C "$WT" status --porcelain=v1 --untracked-files=all | wc -l)"
printf 'FASTWAM_REAL_PARITY_PREFLIGHT_OK\n'
