#!/usr/bin/env bash
set -euo pipefail
R=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
cd "$R"
git apply --check /home/chenyiteng/tmp/scene_fence_build_exports_20260904.patch
git apply /home/chenyiteng/tmp/scene_fence_build_exports_20260904.patch
bash tools/fastwam_scene_fence/build.sh /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin /home/chenyiteng/builds/fastwam-scene-fence-20260904/release
source /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1/runtime/environment.sh
"$VIRTUAL_ENV/bin/python" tools/fastwam_scene_fence/smoke.py --prepare \
 --source-config /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1/runtime/resolved.yaml \
 --output /data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/diagnostics/scene-fence-smoke-20260904
date -Is
