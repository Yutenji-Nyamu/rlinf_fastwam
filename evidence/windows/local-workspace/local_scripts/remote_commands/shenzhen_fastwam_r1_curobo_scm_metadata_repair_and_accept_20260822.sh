#!/usr/bin/env bash
set -euo pipefail

FW=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
PIN=7faa71108368fbb3b6885649f112af607427a2d4
VENDOR="$FW/third_party/RoboTwin"
ENV=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128
PY="$ENV/bin/python"
CACHE_ROOT=/home/chenyiteng/cache/fastwam-7faa
PATCH_RECORD="$CACHE_ROOT/fw-sz-110-patch-record"
FREEZE="$CACHE_ROOT/pip-freeze-after-fw-sz-110.txt"
CUROBO_SRC="$VENDOR/envs/curobo"
CUROBO_REV=d64c4b005459db10c5dd867d8b30a87d5bda9bdb

test -x "$PY"
test "$(git -C "$FW" rev-parse HEAD)" = "$PIN"
test "$(git -C "$CUROBO_SRC" rev-parse HEAD)" = "$CUROBO_REV"
test -d "$PATCH_RECORD"
test ! -e "$FREEZE"

source /etc/profile.d/mihomo-proxy.sh
export PATH="$ENV/bin:/usr/local/cuda/bin:$PATH"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR="$CACHE_ROOT/tmp"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions/fw-sz-110-curobo-v078-sm90"
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
export CUDA_VISIBLE_DEVICES=3
unset PYTHONPATH

printf '%s\n' '=== CuRobo SCM metadata repair ==='
printf 'timestamp_start=%s\n' "$(date --iso-8601=seconds)"
"$PY" -m pip install --index-url https://pypi.org/simple 'setuptools-scm==9.2.2'
"$PY" -m pip install --index-url https://pypi.org/simple \
  --no-build-isolation --no-deps --force-reinstall -e "$CUROBO_SRC"

printf '%s\n' '=== FW-SZ-110 final import and binary-extension acceptance ==='
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
    'torch': '2.7.1+cu128',
    'torchvision': '0.22.1+cu128',
    'numpy': '2.2.6',
    'scipy': '1.15.3',
    'setuptools': '80.9.0',
    'setuptools-scm': '9.2.2',
    'transforms3d': '0.4.2',
    'sapien': '3.0.0b1',
    'mplib': '0.2.1',
    'gymnasium': '0.29.1',
    'trimesh': '4.4.3',
    'open3d': '0.18.0',
    'h5py': '3.16.0',
    'pyglet': '1.5.31',
    'toppra': '0.6.3',
    'opencv-python': '4.11.0.86',
    'PyYAML': '6.0.3',
    'pybind11': '3.0.4',
    'networkx': '3.4.2',
    'numpy-quaternion': '2024.0.13',
    'yourdfpy': '0.0.60',
    'importlib-resources': '6.5.2',
    'scikit-image': '0.25.2',
    'lazy-loader': '0.5',
    'tifffile': '2025.3.13',
    'warp-lang': '1.11.1',
    'nvidia-curobo': '0.7.8',
}
actual = {name: md.version(name) for name in expected}
assert actual == expected, {k: (expected[k], actual[k]) for k in expected if actual[k] != expected[k]}
assert torch.version.cuda == '12.8'
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) == (9, 0)

dist = md.distribution('nvidia-curobo')
direct = json.loads(dist.read_text('direct_url.json'))
editable = Path(urllib.parse.unquote(urllib.parse.urlparse(direct['url']).path)).resolve()
assert direct.get('dir_info', {}).get('editable') is True
assert editable == curobo_src

binary_paths = {}
for module in (
    'curobo.curobolib.lbfgs_step_cu',
    'curobo.curobolib.kinematics_fused_cu',
    'curobo.curobolib.line_search_cu',
    'curobo.curobolib.tensor_step_cu',
    'curobo.curobolib.geom_cu',
):
    loaded = importlib.import_module(module)
    path = Path(loaded.__file__).resolve()
    assert env in path.parents or curobo_src in path.parents
    binary_paths[module] = str(path)

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

wp.init()
assert wp.is_cuda_available()
assert hasattr(wp, 'torch')
_ = wp.torch.device_from_torch(torch.device('cuda:0'))

import envs
from envs.robot.planner import CuroboPlanner, MplibPlanner

print(json.dumps({
    'versions': actual,
    'torch_cuda': torch.version.cuda,
    'visible_gpu': torch.cuda.get_device_name(0),
    'visible_capability': torch.cuda.get_device_capability(0),
    'curobo_source': str(editable),
    'curobo_revision': curobo_rev,
    'binary_extensions': binary_paths,
    'robotwin_eager_imports': 'ok',
    'pytorch3d': 'intentionally_not_installed_for_rgb_qpos_path',
    'lbfgs_workaround': 'not_applied',
}, indent=2))
PY

pip_check_output="$("$PY" -m pip check 2>&1 || true)"
printf '%s\n' "$pip_check_output"
test "$pip_check_output" = 'mplib 0.2.1 has requirement numpy<2.0, but you have numpy 2.2.6.'
"$PY" -m pip freeze | LC_ALL=C sort > "$FREEZE"
sha256sum "$FREEZE" "$PATCH_RECORD"/*
du -sh "$ENV" "$CACHE_ROOT" "$CUROBO_SRC"
git -C "$FW" diff --quiet
git -C "$FW" diff --cached --quiet
git -C "$CUROBO_SRC" status --short --branch
nvidia-smi -i 3,4,5,6,7 --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
printf 'timestamp_end=%s\n' "$(date --iso-8601=seconds)"
printf '%s\n' FASTWAM_FW_SZ_110_OK
