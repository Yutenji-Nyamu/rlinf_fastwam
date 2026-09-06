#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
date -Is
id
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -
printf '\nBUILD_INVENTORY\n'
df -h /home/chenyiteng /data/chenyiteng
for t in cmake ninja g++ gcc patchelf nvcc; do command -v "$t" || true; done
ls -ld /usr/local/cuda* /home/chenyiteng/* /data/chenyiteng/projects/rlinf-shenzhen/*
find /data/chenyiteng/projects /home/chenyiteng/.cache /home/chenyiteng/builds -maxdepth 4 -type d \( -iname '*sapien*' -o -iname '*vulkan*' -o -iname '*oidn*' \) 2>/dev/null || true
base=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo
git -C "$base" branch --show-current
git -C "$base" remote -v
git -C "$base" status --porcelain
find /usr/share/vulkan /etc/vulkan -maxdepth 2 -type f -name '*validation*' 2>/dev/null || true
ls /usr/include/vulkan /usr/include/glm /usr/include/assimp /usr/include/spdlog 2>/dev/null | head -35 || true
date -Is
