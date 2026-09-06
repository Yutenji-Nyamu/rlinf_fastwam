#!/usr/bin/env bash
set -euo pipefail
S=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages
sed -n '1,240p' "$S/sapien/__init__.py"
find "$S/sapien" "$S/sapien.libs" /home/chenyiteng/.sapien /home/chenyiteng/.cache -maxdepth 5 -name 'libOpenImageDenoise*' -print 2>/dev/null || true
ldd /home/chenyiteng/builds/fastwam-scene-fence-20260904/release/librlinf_scene_fence.so
date -Is
