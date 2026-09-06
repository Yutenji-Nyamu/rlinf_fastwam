#!/usr/bin/env bash
set -euo pipefail

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
NATIVE_RT=/data/chenyiteng/projects/robotwin-native/RoboTwin
COMPAT_RT=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
PIN=7faa71108368fbb3b6885649f112af607427a2d4

test "$(git -C "$FW" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$FW" status --porcelain)"

printf '%s\n' '=== FastWAM pyproject ==='
sed -n '1,260p' "$FW/pyproject.toml"

printf '%s\n' '=== FastWAM RoboTwin source contract ==='
sed -n '1,220p' "$FW/third_party/RoboTwin/README.vendor.md"
find "$FW/third_party/RoboTwin" -maxdepth 2 -type f \
  \( -name 'requirements.txt' -o -name '_install.sh' -o -name 'install.sh' -o -name 'README.md' \) \
  -print | sort
for file in \
  "$FW/configs/sim_robotwin.yaml" \
  "$FW/configs/task/robotwin_uncond_3cam_384_1e-4.yaml" \
  "$FW/configs/model/fastwam.yaml"; do
  printf -- '--- %s ---\n' "$file"
  sed -n '1,260p' "$file"
done

printf '%s\n' '=== existing RoboTwin trees ==='
for rt in "$NATIVE_RT" "$COMPAT_RT"; do
  if [[ -d "$rt/.git" ]]; then
    printf -- '--- %s ---\n' "$rt"
    git -C "$rt" rev-parse HEAD
    git -C "$rt" rev-parse HEAD:assets 2>/dev/null || true
    git -C "$rt" rev-parse HEAD:task_config 2>/dev/null || true
    du -sh "$rt/assets" "$rt/task_config" 2>/dev/null || true
    git -C "$rt" status --short -- assets task_config 2>/dev/null || true
  fi
done

printf '%s\n' '=== Conda and CUDA toolchain ==='
export PATH=/home/chenyiteng/miniforge3/bin:$PATH
command -v conda
conda info --base
conda env list
command -v nvcc || true
nvcc --version 2>/dev/null | tail -n 5 || true
ls -ld /usr/local/cuda /usr/local/cuda-* 2>/dev/null || true
printf '%s\n' FASTWAM_R0_CONTRACT_OK
