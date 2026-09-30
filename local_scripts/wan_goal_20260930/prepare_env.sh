#!/bin/bash
set -euo pipefail
ROOT=/data/chenyiteng/projects/wan-goal-sz3
export PATH=/home/chenyiteng/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin:/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin:$PATH
export TMPDIR="$ROOT/tmp" UV_CACHE_DIR="$ROOT/cache/uv-oft" PIP_CACHE_DIR="$ROOT/cache/pip" UV_LINK_MODE=copy
export HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache/xdg"
export UV_PYTHON_INSTALL_DIR="$ROOT/envs/python" OPENPI_DATA_HOME="$ROOT/models/openpi-assets"
export LIBERO_CONFIG_PATH="$ROOT/config/libero" CUDA_VISIBLE_DEVICES=""
export DOWNLOAD_DIR="$ROOT/assets" WAN_PATH="$ROOT/src/diffsynth-studio"
export TORCH_HOME="$ROOT/cache/torch" TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/torchinductor"
export MAX_JOBS=8 UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_BUILDS=4 UV_CONCURRENT_INSTALLS=1
mkdir -p "$LIBERO_CONFIG_PATH" "$OPENPI_DATA_HOME"
mkdir -p "$ROOT/src" "$DOWNLOAD_DIR"
if [ ! -d "$WAN_PATH/.git" ]; then
  git clone https://github.com/RLinf/diffsynth-studio.git "$WAN_PATH"
fi
git -C "$WAN_PATH" checkout 2a2e05fa1f724828b243f272540989b19a6e54f8
cd "$ROOT/RLinf"
test "$(git rev-parse HEAD)" = d34d4c320d08cb982de034aa9a011f08dc0fa217
bash requirements/install.sh embodied --model openvla-oft --env wan --venv "$ROOT/envs/oft-wan" --no-root
uv pip freeze --python "$ROOT/envs/oft-wan/bin/python" > "$ROOT/logs/oft-wan-pip-freeze.txt"
printf 'OFT_ENV_INSTALL_COMPLETE\n'
