#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin

test "$(git -C "$ROOT" rev-parse HEAD)" = 7d07a4212ee6858cc333e1d4fab7a37256d1f839
test -z "$(git -C "$ROOT" status --porcelain)"
test ! -e "$VENV"
test -x /home/chenyiteng/miniforge3/bin/pip
test -x /home/chenyiteng/miniforge3/envs/act/bin/python

printf '%s\n' '=== COMMANDS ==='
for command_name in gcc g++ cmake patchelf wget curl unzip git-lfs ffmpeg; do
  command -v "$command_name" || printf 'MISSING %s\n' "$command_name"
done
test -x /home/chenyiteng/miniforge3/envs/act/bin/patchelf && \
  printf 'ACT_ENV_PATCHELF=%s\n' /home/chenyiteng/miniforge3/envs/act/bin/patchelf || true

printf '%s\n' '=== CUDA/VULKAN ==='
readlink -f /usr/local/cuda || true
if test -x /usr/local/cuda/bin/nvcc; then
  /usr/local/cuda/bin/nvcc --version
else
  printf '%s\n' 'MISSING /usr/local/cuda/bin/nvcc'
fi
for header in cusparse.h cublas_v2.h cusolverDn.h; do
  if test -s "/usr/local/cuda/include/$header"; then
    stat -c '%s %n' "/usr/local/cuda/include/$header"
  else
    printf 'MISSING %s\n' "/usr/local/cuda/include/$header"
  fi
done
if test -s /etc/vulkan/icd.d/nvidia_icd.json; then
  stat -c '%s %n' /etc/vulkan/icd.d/nvidia_icd.json
else
  printf '%s\n' 'MISSING /etc/vulkan/icd.d/nvidia_icd.json'
fi

printf '%s\n' '=== DYNAMIC LIBS ==='
ldconfig -p | grep -E 'lib(EGL|GL|vulkan|OSMesa|avcodec|avformat|avutil)\.so' | head -n 80 || true

printf '%s\n' '=== SPACE ==='
df -h / /home /data
printf '%s\n' 'R1_SYSTEM_DEP_PREFLIGHT_OK'
