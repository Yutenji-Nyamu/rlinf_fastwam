#!/bin/bash
set -euo pipefail
ROOT=/data/chenyiteng/projects/wan-goal-sz3
export PATH=/home/chenyiteng/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin:/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917/bin:$PATH
export TMPDIR="$ROOT/tmp" UV_CACHE_DIR="$ROOT/cache/uv" PIP_CACHE_DIR="$ROOT/cache/pip" UV_LINK_MODE=copy
export HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache/xdg"
export UV_PYTHON_INSTALL_DIR="$ROOT/envs/python" OPENPI_DATA_HOME="$ROOT/models/openpi-assets"
export LIBERO_CONFIG_PATH="$ROOT/config/libero-pi05" CUDA_VISIBLE_DEVICES=""
export DOWNLOAD_DIR="$ROOT/assets" WAN_PATH="$ROOT/src/diffsynth-studio"
export TORCH_HOME="$ROOT/cache/torch" TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/torchinductor"
export MAX_JOBS=8 UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_BUILDS=4 UV_CONCURRENT_INSTALLS=1
mkdir -p "$LIBERO_CONFIG_PATH" "$OPENPI_DATA_HOME"
if [ ! -d "$ROOT/RLinf-pi05/.git" ]; then
  git clone --no-hardlinks "$ROOT/RLinf" "$ROOT/RLinf-pi05"
fi
cd "$ROOT/RLinf-pi05"
test "$(git rev-parse HEAD)" = d34d4c320d08cb982de034aa9a011f08dc0fa217
bash requirements/install.sh embodied --model openpi --env libero --venv "$ROOT/envs/pi05-wan" --no-root
source "$ROOT/envs/pi05-wan/bin/activate"
uv pip install -e "$WAN_PATH"
uv pip install -r requirements/embodied/models/wan.txt
# OpenPI's optional ALOHA dependency can pull a newer dm-control/MuJoCo pair
# after rlinf-libero was installed. Keep LIBERO's MuJoCo <3.4 requirement.
# This exact compatible pair is also used by the pinned official installer.
uv pip install --no-deps "dm-control==1.0.34" "mujoco==3.3.7"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id='RLinf/openpi_tokenizer',revision='befaa248e4f82954b625a421658f933dfd1a97a0',local_dir=os.environ['OPENPI_DATA_HOME'])
PY
uv pip freeze > "$ROOT/logs/pi05-wan-pip-freeze.txt"
printf 'PI05_ENV_INSTALL_COMPLETE\n'
