#!/usr/bin/env bash
set -euo pipefail

repo=/data/chenyiteng/projects/fasterwam-standalone/FasterWAM-hustvl-official
venv="$repo/.venvs/robotwin"
curobo="$repo/third_party/RoboTwin/envs/curobo"
py_include=/home/chenyiteng/miniforge3/envs/RoboTwin/include/python3.10
uv_cache=/data/chenyiteng/cache/uv-fasterwam

test "$(git -C "$repo" rev-parse HEAD)" = 83667817df0d4f823f39d90700e61ea2f432ac45
test "$(git -C "$curobo" rev-parse HEAD)" = d64c4b005459db10c5dd867d8b30a87d5bda9bdb
test -x "$venv/bin/python"
test -f "$py_include/Python.h"

gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
if [ "$gpu_mem" -ge 100 ]; then
  echo "GPU3 is no longer free: ${gpu_mem} MiB" >&2
  exit 20
fi

# Use the official installer's resolver because non-interactive SSH does not
# necessarily expose uv on PATH.
FASTERWAM_ROOT="$repo"
export PATH=/home/chenyiteng/miniforge3/bin:$PATH
source "$repo/scripts/setup/_common.sh"
uv_bin=$(fasterwam_find_uv)
echo "UV=$uv_bin"
echo "PYTHON_HEADER=$py_include/Python.h"
echo "GPU3_BEFORE_MIB=$gpu_mem"

cd "$repo"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  CUDA_VISIBLE_DEVICES=3 \
  TORCH_CUDA_ARCH_LIST=9.0 \
  MAX_JOBS=8 \
  CMAKE_BUILD_PARALLEL_LEVEL=8 \
  CPATH="$py_include${CPATH:+:$CPATH}" \
  CPLUS_INCLUDE_PATH="$py_include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" \
  UV_CACHE_DIR="$uv_cache" \
  "$uv_bin" pip install --python "$venv/bin/python" \
    --no-deps --no-build-isolation -e "$curobo"

# These are the only official installer steps after the CuRobo editable install.
test -d third_party/RoboTwin/assets/background_texture
test -d third_party/RoboTwin/assets/embodiments
test -d third_party/RoboTwin/assets/objects
ln -sfn "$repo/experiments/robotwin/fasterwam_policy" \
  "$repo/third_party/RoboTwin/policy/fasterwam_policy"

"$venv/bin/python" - <<'PY'
import numpy
import torch
import sapien
import curobo
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("sapien", getattr(sapien, "__version__", "import-ok"))
print("curobo", getattr(curobo, "__version__", "import-ok"))
PY

echo "POLICY_LINK=$(readlink -f third_party/RoboTwin/policy/fasterwam_policy)"
du -sh "$venv" "$uv_cache" "$curobo" | sort -h
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 3
