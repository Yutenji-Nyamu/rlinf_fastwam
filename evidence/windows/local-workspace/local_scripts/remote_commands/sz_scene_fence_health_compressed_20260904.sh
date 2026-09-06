#!/usr/bin/env bash
set -euo pipefail
id
export FASTWAM_RUN_PATH=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3
export PYTHONDONTWRITEBYTECODE=1
export FASTWAM_DRIVER_PID=1568973
export SNAPSHOT_OUTPUT_COMPRESSED=1
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -
sha256sum /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/sapien.libs/libsvulkan2.so /home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/librlinf_scene_fence.so
