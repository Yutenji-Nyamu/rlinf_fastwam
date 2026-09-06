#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '=== PATCHELF CANDIDATES ==='
find /home/chenyiteng/miniforge3 -xdev -type f -name patchelf -perm -u+x -print 2>/dev/null | head -n 30 || true

printf '%s\n' '=== OFFICIAL APT PACKAGE STATUS ==='
for package_name in \
  linux-libc-dev build-essential wget unzip curl cmake patchelf git-lfs lsb-release \
  libavutil-dev libavcodec-dev libavformat-dev libavfilter-dev libavdevice-dev \
  libibverbs-dev ncurses-term mesa-utils libosmesa6-dev freeglut3-dev libglew-dev \
  libegl1 libgles2 libglvnd-dev libglfw3-dev libgl1-mesa-dev libglib2.0-0 libsm6 \
  libxext6 libxrender-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
  libaio-dev libgomp1 libexpat1 libfontconfig1-dev libpython3-stdlib imagemagick \
  libmagickwand-dev libvulkan1 vulkan-tools libnuma1 mesa-vulkan-drivers; do
  status=$(dpkg-query -W -f='${db:Status-Status}' "$package_name" 2>/dev/null || true)
  if test "$status" = 'installed'; then
    printf 'INSTALLED %s\n' "$package_name"
  else
    printf 'MISSING %s\n' "$package_name"
  fi
done
