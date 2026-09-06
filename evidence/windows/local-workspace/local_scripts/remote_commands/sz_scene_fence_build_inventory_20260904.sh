#!/usr/bin/env bash
set -euo pipefail
date -Is
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
find "$venv" -maxdepth 6 -type d \( -name 'include' -o -name 'headers' -o -name 'svulkan2' \) | head -60
ls "$venv/bin" | grep -E 'cmake|ninja|patchelf' || true
find "$venv/lib/python3.11/site-packages/sapien-3.0.1.data" -maxdepth 4 -type d 2>/dev/null | head -35 || true
readelf -d "$venv/lib/python3.11/site-packages/sapien.libs/libsvulkan2.so" | grep -E 'NEEDED|RPATH|RUNPATH|SONAME'
readelf -d "$venv/lib/python3.11/site-packages/sapien/pysapien.cpython-311-x86_64-linux-gnu.so" | grep -E 'NEEDED|RPATH|RUNPATH'
dpkg-query -W 'libx11-dev' 'libxrandr-dev' 'libxinerama-dev' 'libxcursor-dev' 'libxi-dev' 'libwayland-dev' 'libvulkan-dev' 2>/dev/null || true
cmake --version | head -1
g++ --version | head -1
du -sh "$venv/lib/python3.11/site-packages/sapien" "$venv/lib/python3.11/site-packages/sapien.libs"
git -C /data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo ls-files | grep -E 'fastwam.*(sh|test)|tools/.*fastwam' | head -40
date -Is
