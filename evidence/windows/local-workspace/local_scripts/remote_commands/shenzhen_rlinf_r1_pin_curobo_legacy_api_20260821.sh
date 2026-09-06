#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RUN=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821
CUROBO_REV=d64c4b005459db10c5dd867d8b30a87d5bda9bdb

source "$VENV/bin/activate"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST='7.0;8.0;9.0;10.0'
export MAX_JOBS=16

exec > >(tee "$RUN/curobo_v078_pin.log") 2>&1

printf '%s\n' '=== PIN CUROBO TO LAST LEGACY API RELEASE ==='
timeout --signal=INT --kill-after=120s 1800s \
  uv pip install --python "$VENV/bin/python" --reinstall \
    "git+https://github.com/NVlabs/curobo.git@$CUROBO_REV" \
    --no-build-isolation

"$VENV/bin/python" - "$CUROBO_REV" <<'PY'
from __future__ import annotations

import importlib.metadata as md
import json
from pathlib import Path
import sys

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState

dist = md.distribution("nvidia-curobo")
direct = json.loads((Path(dist._path) / "direct_url.json").read_text())
assert direct["vcs_info"]["commit_id"] == sys.argv[1], direct
print({
    "version": dist.version,
    "commit": direct["vcs_info"]["commit_id"],
    "TensorDeviceType": str(TensorDeviceType),
    "Pose": str(Pose),
    "JointState": str(JointState),
})
PY

printf '%s\n' 'R1_CUROBO_V078_PIN_OK'
