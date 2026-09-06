#!/usr/bin/env bash
set -euxo pipefail
ROOT=/root/autodl-tmp
ENV="$ROOT/conda/envs/FastWAM-RLinf"
FW="$ROOT/FastWAM"
CUROBO_ORIG="$FW/third_party/RoboTwin/envs/curobo"
CUROBO_SRC="$ROOT/src/curobo-fastwam-rlinf-v0.7.8"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"
unset OMP_NUM_THREADS
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
export PIP_CACHE_DIR="$ROOT/cache/pip"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions/fastwam-rlinf-torch271-cu128"
export TMPDIR="$ROOT/tmp/fastwam-rlinf"
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST=8.0

python -m pip install \
  transforms3d==0.4.2 sapien==3.0.0b1 scipy==1.15.3 mplib==0.2.1 \
  gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 h5py==3.16.0 \
  pyglet==1.5.31 toppra==0.6.3 opencv-python==4.11.0.86 \
  opencv-python-headless==4.11.0.86 matplotlib==3.10.9 PyYAML==6.0.3 \
  warp-lang==1.11.1

mkdir -p "$ROOT/src"
if [ ! -d "$CUROBO_SRC/.git" ]; then
  git clone --local --no-hardlinks "$CUROBO_ORIG" "$CUROBO_SRC"
fi
test "$(git -C "$CUROBO_SRC" rev-parse HEAD)" = d64c4b005459db10c5dd867d8b30a87d5bda9bdb
python -m pip install -e "$CUROBO_SRC" --no-build-isolation

SAPIEN_LOCATION="$(python -m pip show sapien | awk '/^Location:/{print $2}')/sapien"
MPLIB_LOCATION="$(python -m pip show mplib | awk '/^Location:/{print $2}')/mplib"
export URDF_LOADER="$SAPIEN_LOCATION/wrapper/urdf_loader.py"
export PLANNER="$MPLIB_LOCATION/planner.py"
test -f "$URDF_LOADER"
test -f "$PLANNER"
python - <<'PY'
import os
from pathlib import Path


def replace_once_or_accept(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected patch source not found in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


urdf_loader = Path(os.environ["URDF_LOADER"])
replace_once_or_accept(
    urdf_loader,
    'with open(urdf_file, "r") as f:',
    'with open(urdf_file, "r", encoding="utf-8") as f:',
)
replace_once_or_accept(
    urdf_loader,
    'with open(srdf_file, "r") as f:',
    'with open(srdf_file, "r", encoding="utf-8") as f:',
)
replace_once_or_accept(
    Path(os.environ["PLANNER"]),
    "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
    "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
)
PY

python -m py_compile "$URDF_LOADER" "$PLANNER"
python -m pip install "setuptools==80.9.0" "warp-lang==1.11.1"
python -m pip check
python - <<'PY'
import sys
from importlib.metadata import version

import torch
import torchvision

print("python", sys.version.split()[0])
print("torch", torch.__version__, torch.version.cuda)
print("torchvision", torchvision.__version__)
for package in (
    "fastwam",
    "rlinf",
    "transformers",
    "hydra-core",
    "omegaconf",
    "numpy",
    "torchcodec",
    "setuptools",
    "warp-lang",
    "ray",
    "toppra",
    "nvidia_curobo",
):
    print(package, version(package))
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
touch /root/autodl-tmp/fastwam-rlinf-setup/install_joint_env.done
