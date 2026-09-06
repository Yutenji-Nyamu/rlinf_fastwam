#!/usr/bin/env bash
set -euo pipefail

# FW-SZ-110: install only the vendored RoboTwin runtime surface needed by
# FastWAM's evaluator, then build source-locked CuRobo in the standalone env.
# This packet does not copy assets/models, start SAPIEN, run an expert, or run
# FastWAM inference.  It deliberately does not carry over the native-ACT
# Hopper/LBFGS compatibility patch.

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
PIN=7faa71108368fbb3b6885649f112af607427a2d4
OFFICIAL_REMOTE=https://github.com/yuantianyuan01/FastWAM.git
VENDOR="$FW/third_party/RoboTwin"
VENDOR_TREE=b0c3bd309d95da41224191a5b089d94727317315
VENDOR_UPSTREAM=bf44be51cf5717a5595ce59447f2cf5263d2aa95
VENDOR_INSTALL_BLOB=8609fbb7ec88f2d4471f715674f08b6b14c3a36c
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
PY="$ENV/bin/python"
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa
BASE_FREEZE="$CACHE_ROOT/pip-freeze-after-fastwam.txt"
PATCH_RECORD="$CACHE_ROOT/fw-sz-110-patch-record"
FREEZE="$CACHE_ROOT/pip-freeze-after-fw-sz-110.txt"
CUROBO_URL=https://github.com/NVlabs/curobo.git
CUROBO_TAG=v0.7.8
CUROBO_REV=d64c4b005459db10c5dd867d8b30a87d5bda9bdb
CUROBO_SRC="$VENDOR/envs/curobo"

printf 'timestamp_start=%s\n' "$(date --iso-8601=seconds)"
printf 'source=%s\nenv=%s\nvendor=%s\ncurobo_source=%s\n' \
  "$FW" "$ENV" "$VENDOR" "$CUROBO_SRC"

printf '%s\n' '=== exact standalone/source preconditions ==='
test -x "$PY"
test -f "$BASE_FREEZE"
test "$(git -C "$FW" rev-parse HEAD)" = "$PIN"
test "$(git -C "$FW" remote get-url origin)" = "$OFFICIAL_REMOTE"
test "$(git -C "$FW" rev-parse "$PIN:third_party/RoboTwin")" = "$VENDOR_TREE"
test "$(git -C "$FW" hash-object "$VENDOR/script/_install.sh")" = "$VENDOR_INSTALL_BLOB"
test -z "$(git -C "$FW" status --porcelain)"
test ! -e "$VENDOR/script/requirements.txt"
if git -C "$FW" cat-file -e "$PIN:third_party/RoboTwin/script/requirements.txt" 2>/dev/null; then
  echo 'STOP: unexpected vendored requirements.txt; re-audit the source before proceeding'
  exit 1
fi
test ! -e "$CUROBO_SRC"
test ! -e "$PATCH_RECORD"
test ! -e "$FREEZE"

export PATH="$ENV/bin:$PATH"
unset PYTHONPATH
test "$(command -v python)" = "$PY"

"$PY" - "$FW" <<'PY'
from __future__ import annotations

import importlib.metadata as md
import json
from pathlib import Path
import sys
import urllib.parse

import fastwam
import numpy
import torch
import torchvision

fw = Path(sys.argv[1]).resolve()
direct = json.loads(md.distribution("fastwam").read_text("direct_url.json"))
editable = Path(
    urllib.parse.unquote(urllib.parse.urlparse(direct["url"]).path)
).resolve()
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torchvision.__version__ == "0.22.1+cu128", torchvision.__version__
assert numpy.__version__ == "2.2.6", numpy.__version__
assert md.version("fastwam") == "0.1.0", md.version("fastwam")
assert direct.get("dir_info", {}).get("editable") is True, direct
assert editable == fw, (editable, fw)
assert fw in Path(fastwam.__file__).resolve().parents
print({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "fastwam": md.version("fastwam"),
    "editable_root": str(editable),
})
PY

printf '%s\n' '=== resource and PPO-boundary snapshot (read only) ==='
df -h /home /data
free -h
nvidia-smi -i 3 --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
gpu3_apps="$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | tr -d '[:space:]')"
test -z "$gpu3_apps"
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

# Proxy and all caches are process-local.  Nothing is written to shell, pip,
# or Git configuration, and no RLinf environment is activated or referenced.
source /etc/profile.d/mihomo-proxy.sh
mkdir -p \
  "$CACHE_ROOT/pip" \
  "$CACHE_ROOT/huggingface/hub" \
  "$CACHE_ROOT/modelscope" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/torch_extensions/fw-sz-110-curobo-v078-sm90" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/cuda" \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/tmp"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export MODELSCOPE_CACHE="$CACHE_ROOT/modelscope"
export TORCH_HOME="$CACHE_ROOT/torch"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions/fw-sz-110-curobo-v078-sm90"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"
export PIP_DEFAULT_TIMEOUT=120
export CUDA_VISIBLE_DEVICES=3
unset PIP_TRUSTED_HOST

printf '%s\n' '=== process-only network route ==='
env | grep -iE '^(http|https|all|no)_proxy=' \
  | sed -E 's#(https?://)[^/@]+@#\1REDACTED@#I' \
  | sort
curl -sSIL --max-time 20 --connect-timeout 8 -o /dev/null \
  -w 'pypi http=%{http_code} remote=%{remote_ip} total=%{time_total}\n' \
  https://pypi.org/simple/
curl -sSIL --max-time 20 --connect-timeout 8 -o /dev/null \
  -w 'github http=%{http_code} remote=%{remote_ip} total=%{time_total}\n' \
  https://github.com/NVlabs/curobo

printf '%s\n' '=== actual CUDA extension toolchain ==='
NVCC_BIN=/usr/local/cuda/bin/nvcc
test -x "$NVCC_BIN"
NVCC_REAL="$(readlink -f "$NVCC_BIN")"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_REAL")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
NVCC_RELEASE="$("$CUDA_HOME/bin/nvcc" --version \
  | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' \
  | head -n 1)"
test "$NVCC_RELEASE" = 12.9
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
printf 'nvcc_bin=%s\nnvcc_real=%s\nCUDA_HOME=%s\nnvcc_release=%s\nTORCH_CUDA_ARCH_LIST=%s\nMAX_JOBS=%s\n' \
  "$NVCC_BIN" "$NVCC_REAL" "$CUDA_HOME" "$NVCC_RELEASE" \
  "$TORCH_CUDA_ARCH_LIST" "$MAX_JOBS"
"$CUDA_HOME/bin/nvcc" --version
gcc --version | head -n 1
g++ --version | head -n 1
ninja --version

"$PY" - "$CUDA_HOME" <<'PY'
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import CUDA_HOME

expected = Path(sys.argv[1]).resolve()
actual = Path(CUDA_HOME).resolve() if CUDA_HOME else None
assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert actual == expected, (actual, expected)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
assert torch.cuda.get_device_capability(0) == (9, 0), torch.cuda.get_device_capability(0)
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cpp_extension_cuda_home": str(actual),
    "visible_gpu": torch.cuda.get_device_name(0),
    "visible_capability": torch.cuda.get_device_capability(0),
})
PY

printf '%s\n' '=== minimal source-audited RoboTwin/CuRobo dependencies ==='
# Do not execute the vendored _install.sh: its requirements file is absent at
# the locked FastWAM commit, while the exact upstream file pins torch==2.4.1
# and huggingface_hub==0.25.0.  The list below covers eager imports in
# envs/_base_task.py, envs/utils, envs/camera, envs/robot/planner.py, plus the
# declared runtime dependencies of source-locked CuRobo.  PyTorch3D is omitted:
# camera.py catches its absence and FastWAM RGB+qpos evaluation does not call
# the optional farthest-point-sampling helper.
# MPlib 0.2.1 is the only official release and RoboTwin's exact pin.  Its wheel
# metadata predates NumPy 2 and declares numpy<2.0, while the current Fast-WAM
# source pins numpy==2.2.6.  Install the exact wheel without dependency
# resolution, keep Fast-WAM's NumPy, and prove the imported runtime below.
"$PY" -m pip install --index-url https://pypi.org/simple --no-deps 'mplib==0.2.1'

"$PY" -m pip install --index-url https://pypi.org/simple \
  'setuptools==80.9.0' \
  'numpy==2.2.6' \
  'scipy==1.15.3' \
  'transforms3d==0.4.2' \
  'sapien==3.0.0b1' \
  'gymnasium==0.29.1' \
  'trimesh==4.4.3' \
  'open3d==0.18.0' \
  'h5py==3.16.0' \
  'pyglet==1.5.31' \
  'toppra==0.6.3' \
  'opencv-python==4.11.0.86' \
  'PyYAML==6.0.3' \
  'pybind11==3.0.4' \
  'networkx==3.4.2' \
  'numpy-quaternion==2024.0.13' \
  'yourdfpy==0.0.60' \
  'importlib-resources==6.5.2' \
  'scikit-image==0.25.2' \
  'lazy-loader==0.5' \
  'tifffile==2025.3.13' \
  'warp-lang==1.11.1'

command -v ffmpeg
ffmpeg -version | head -n 1

printf '%s\n' '=== exact official SAPIEN/MPLib compatibility semantics ==='
SAPIEN_ROOT="$("$PY" - <<'PY'
from pathlib import Path
import sapien
print(Path(sapien.__file__).resolve().parent)
PY
)"
MPLIB_ROOT="$("$PY" - <<'PY'
from pathlib import Path
import mplib
print(Path(mplib.__file__).resolve().parent)
PY
)"
URDF_LOADER="$SAPIEN_ROOT/wrapper/urdf_loader.py"
MPLIB_PLANNER="$MPLIB_ROOT/planner.py"
test -f "$URDF_LOADER"
test -f "$MPLIB_PLANNER"
case "$(readlink -f "$URDF_LOADER")" in "$ENV"/*) ;; *) echo 'STOP: SAPIEN is outside FastWAM env'; exit 1 ;; esac
case "$(readlink -f "$MPLIB_PLANNER")" in "$ENV"/*) ;; *) echo 'STOP: MPLib is outside FastWAM env'; exit 1 ;; esac
mkdir -p "$PATCH_RECORD"
printf 'urdf_loader=%s\nmplib_planner=%s\n' "$URDF_LOADER" "$MPLIB_PLANNER" \
  > "$PATCH_RECORD/paths.txt"
sha256sum "$URDF_LOADER" "$MPLIB_PLANNER" > "$PATCH_RECORD/before.sha256"
grep -nE 'with open\((urdf_file|srdf_file)' "$URDF_LOADER" \
  > "$PATCH_RECORD/before.lines.txt" || true
grep -nF 'if np.linalg.norm(delta_twist) < 1e-4' "$MPLIB_PLANNER" \
  >> "$PATCH_RECORD/before.lines.txt" || true
cat "$PATCH_RECORD/before.sha256"
cat "$PATCH_RECORD/before.lines.txt"

URDF_LOADER="$URDF_LOADER" MPLIB_PLANNER="$MPLIB_PLANNER" "$PY" - <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def replace_once_or_accept(path: Path, old: str, new: str) -> str:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return "already_correct"
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one patch source in {path}: count={count}, old={old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    return "patched"


urdf = Path(os.environ["URDF_LOADER"])
planner = Path(os.environ["MPLIB_PLANNER"])
results = {
    "urdf_utf8": replace_once_or_accept(
        urdf,
        'with open(urdf_file, "r") as f:',
        'with open(urdf_file, "r", encoding="utf-8") as f:',
    ),
    "srdf_utf8": replace_once_or_accept(
        urdf,
        'with open(srdf_file, "r") as f:',
        'with open(srdf_file, "r", encoding="utf-8") as f:',
    ),
    "mplib_remove_collide": replace_once_or_accept(
        planner,
        "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
        "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
    ),
}
for path in (urdf, planner):
    data = path.read_bytes()
    bad = sorted({byte for byte in data if byte < 32 and byte not in (9, 10, 13)})
    if bad:
        raise RuntimeError(f"control bytes in {path}: {bad}")
    print(path, hashlib.sha256(data).hexdigest())
print(results)
PY

"$PY" -m py_compile "$URDF_LOADER" "$MPLIB_PLANNER"
sha256sum "$URDF_LOADER" "$MPLIB_PLANNER" > "$PATCH_RECORD/after.sha256"
grep -nE 'with open\((urdf_file|srdf_file)' "$URDF_LOADER" \
  > "$PATCH_RECORD/after.lines.txt"
grep -nF 'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:' "$MPLIB_PLANNER" \
  >> "$PATCH_RECORD/after.lines.txt"
cat "$PATCH_RECORD/after.sha256"
cat "$PATCH_RECORD/after.lines.txt"

printf '%s\n' '=== source-locked CuRobo v0.7.8 build ==='
git clone --branch "$CUROBO_TAG" --depth 1 "$CUROBO_URL" "$CUROBO_SRC"
test "$(git -C "$CUROBO_SRC" rev-parse HEAD)" = "$CUROBO_REV"
test -z "$(git -C "$CUROBO_SRC" status --porcelain)"
printf 'curobo_origin=%s\ncurobo_head=%s\ncurobo_describe=%s\n' \
  "$(git -C "$CUROBO_SRC" remote get-url origin)" \
  "$(git -C "$CUROBO_SRC" rev-parse HEAD)" \
  "$(git -C "$CUROBO_SRC" describe --tags --always --dirty)"

# Dependencies are installed explicitly above, so --no-deps prevents this
# older package's open lower bounds from moving Torch, NumPy, SciPy, or Warp.
"$PY" -m pip install --no-build-isolation --no-deps -e "$CUROBO_SRC"

printf '%s\n' '=== FW-SZ-110 import and binary-extension acceptance ==='
cd "$VENDOR"
"$PY" - "$ENV" "$CUROBO_SRC" "$CUROBO_REV" <<'PY'
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import urllib.parse

import cv2
import gymnasium
import h5py
import mplib
import numpy
import open3d
import sapien
import scipy
import torch
import toppra
import transforms3d
import trimesh
import warp as wp
import yaml

env = Path(sys.argv[1]).resolve()
curobo_src = Path(sys.argv[2]).resolve()
curobo_rev = sys.argv[3]

expected = {
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "setuptools": "80.9.0",
    "transforms3d": "0.4.2",
    "sapien": "3.0.0b1",
    "mplib": "0.2.1",
    "gymnasium": "0.29.1",
    "trimesh": "4.4.3",
    "open3d": "0.18.0",
    "h5py": "3.16.0",
    "pyglet": "1.5.31",
    "toppra": "0.6.3",
    "opencv-python": "4.11.0.86",
    "PyYAML": "6.0.3",
    "pybind11": "3.0.4",
    "networkx": "3.4.2",
    "numpy-quaternion": "2024.0.13",
    "yourdfpy": "0.0.60",
    "importlib-resources": "6.5.2",
    "scikit-image": "0.25.2",
    "lazy-loader": "0.5",
    "tifffile": "2025.3.13",
    "warp-lang": "1.11.1",
    "nvidia-curobo": "0.7.8",
}
actual = {name: md.version(name) for name in expected}
assert actual == expected, {k: (expected[k], actual[k]) for k in expected if actual[k] != expected[k]}
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
assert torch.cuda.get_device_capability(0) == (9, 0), torch.cuda.get_device_capability(0)

dist = md.distribution("nvidia-curobo")
direct = json.loads(dist.read_text("direct_url.json"))
editable = Path(
    urllib.parse.unquote(urllib.parse.urlparse(direct["url"]).path)
).resolve()
assert direct.get("dir_info", {}).get("editable") is True, direct
assert editable == curobo_src, (editable, curobo_src)

for module in (
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
):
    loaded = importlib.import_module(module)
    assert env in Path(loaded.__file__).resolve().parents or curobo_src in Path(loaded.__file__).resolve().parents

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

wp.init()
assert wp.is_cuda_available()
assert hasattr(wp, "torch")
_ = wp.torch.device_from_torch(torch.device("cuda:0"))

# This imports the exact eager surface used by RoboTwin's evaluator, but does
# not instantiate SAPIEN or run any planner/kernel/simulation.
import envs
from envs.robot.planner import CuroboPlanner, MplibPlanner

print(json.dumps({
    "versions": actual,
    "torch_cuda": torch.version.cuda,
    "visible_gpu": torch.cuda.get_device_name(0),
    "visible_capability": torch.cuda.get_device_capability(0),
    "curobo_source": str(editable),
    "curobo_revision": curobo_rev,
    "binary_extensions": "imported_only",
    "robotwin_eager_imports": "ok",
    "pytorch3d": "intentionally_not_installed_for_rgb_qpos_path",
    "lbfgs_workaround": "not_applied",
}, indent=2))
PY

pip_check_output="$("$PY" -m pip check 2>&1 || true)"
printf '%s\n' "$pip_check_output"
test "$pip_check_output" = 'mplib 0.2.1 has requirement numpy<2.0, but you have numpy 2.2.6.'
"$PY" -m pip freeze | LC_ALL=C sort > "$FREEZE"
sha256sum "$FREEZE" "$PATCH_RECORD"/*
du -sh "$ENV" "$CACHE_ROOT" "$CUROBO_SRC"

# A nested, source-locked CuRobo checkout is the only expected new source-tree
# entry.  FastWAM tracked content must remain untouched.
git -C "$FW" diff --quiet
git -C "$FW" diff --cached --quiet
git -C "$FW" status --short --branch
test "$(git -C "$CUROBO_SRC" rev-parse HEAD)" = "$CUROBO_REV"
test -z "$(git -C "$CUROBO_SRC" status --porcelain)"

printf '%s\n' '=== postflight PPO boundary (read only) ==='
nvidia-smi -i 3 --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
printf 'vendor_upstream=%s\ncurobo_revision=%s\n' "$VENDOR_UPSTREAM" "$CUROBO_REV"
printf 'timestamp_end=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' FASTWAM_FW_SZ_110_OK
