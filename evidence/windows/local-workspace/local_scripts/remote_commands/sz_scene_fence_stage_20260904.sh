#!/usr/bin/env bash
set -euo pipefail
R=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
test "$(id -un)" = chenyiteng
test "$(git -C "$R" rev-parse HEAD)" = 4faade1d50bf21d1caf1b8a4e5f89282a810208a
test -z "$(git -C "$R" status --porcelain)"
test ! -e "$R/tools/fastwam_scene_fence"
mkdir -p "$R/tools/fastwam_scene_fence"
cat /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1/runtime/environment.sh
sed -n '380,430p' "$R/rlinf/scheduler/cluster/node.py"
sed -n '130,160p' "$R/rlinf/scheduler/cluster/cluster.py"
date -Is
