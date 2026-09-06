#!/usr/bin/env bash
set -eu

RLINF_ENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
FASTWAM_SRC=/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src

export PYTHONPATH="${FASTWAM_SRC}:${PYTHONPATH:-}"
"${RLINF_ENV}/bin/python" - <<'PY'
import importlib
import sys

print("python", sys.version.split()[0])
for name in (
    "torch",
    "fastwam",
    "hydra",
    "einops",
    "safetensors",
    "diffusers",
    "transformers",
    "imageio",
    "cv2",
    "fastwam.models.wan22.fastwam",
    "fastwam.models.wan22.mot",
    "fastwam.models.wan22.action_dit",
):
    try:
        module = importlib.import_module(name)
        print("ok", name, getattr(module, "__version__", ""))
    except Exception as exc:
        print("fail", name, type(exc).__name__, str(exc))
PY
