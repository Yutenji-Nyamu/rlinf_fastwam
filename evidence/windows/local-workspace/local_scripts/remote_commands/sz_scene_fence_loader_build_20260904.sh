#!/usr/bin/env bash
set -euo pipefail
cd /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
git apply --check /home/chenyiteng/tmp/scene_fence_build_loader_20260904.patch
git apply /home/chenyiteng/tmp/scene_fence_build_loader_20260904.patch
bash tools/fastwam_scene_fence/build.sh /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin /home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final
date -Is
