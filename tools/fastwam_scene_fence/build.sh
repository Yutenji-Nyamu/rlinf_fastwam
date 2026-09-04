#!/usr/bin/env bash
set -euo pipefail
# Explicit arguments; never installs into or changes the shared environment.
venv=${1:?usage: build.sh VENV NEW_OUTPUT_DIR}
out=${2:?usage: build.sh VENV NEW_OUTPUT_DIR}
here=$(cd -- "$(dirname -- "$0")" && pwd)
site=$venv/lib/python3.11/site-packages
base=$site/sapien.libs/libsvulkan2.so
test "$(sha256sum "$base" | cut -d' ' -f1)" = 972eff8fc5fedfd59d4cd8794039d9bc850847a6c74dfbf7d549eff9d041b1b7
test ! -e "$out"
mkdir -p "$out"
# The current wheel uses the C++11 string ABI and CUDA interop class layout.
g++ -std=c++20 -O2 -DNDEBUG -fPIC -shared -fvisibility=hidden \
  -D_GLIBCXX_USE_CXX11_ABI=1 -DSVULKAN2_CUDA_INTEROP -DVK_NO_PROTOTYPES \
  -DVULKAN_HPP_STORAGE_SHARED -DNVTX_DISABLE=1 \
  -I"$site/sapien/include" "$here/scene_fence.cpp" \
  -Wl,--no-as-needed "$base" -Wl,--as-needed \
  -Wl,-z,defs -Wl,-rpath,"$site/sapien.libs" \
  -Wl,--version-script="$here/exports.map" \
  -Wl,-soname,librlinf_scene_fence.so \
  -o "$out/librlinf_scene_fence.so"
sha256sum "$base" "$out/librlinf_scene_fence.so" \
  "$here/scene_fence.cpp" "$site/sapien/include/svulkan2/renderer/rt_renderer.h" \
  > "$out/build.sha256"
g++ --version > "$out/compiler.txt"
nm -D --defined-only "$out/librlinf_scene_fence.so" > "$out/exports.txt"
objdump -d -C "$out/librlinf_scene_fence.so" > "$out/disassembly.txt"
printf 'export RLINF_SCENE_FENCE_LIBRARY=%q\n' "$out/librlinf_scene_fence.so" > "$out/enable.sh"
cat "$out/build.sha256"
cat "$out/exports.txt"
