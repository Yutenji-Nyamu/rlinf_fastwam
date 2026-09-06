#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mihomo-proxy.sh
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
RUN=/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821
CUROBO_OLD=8e734f3ced1df898990bcd92de40abce475907db
CUROBO_REV=d64c4b005459db10c5dd867d8b30a87d5bda9bdb

source "$VENV/bin/activate"
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST='7.0;8.0;9.0;10.0'
export MAX_JOBS=16

exec > >(tee "$RUN/curobo_v078_pin_no_deps_retry1.log") 2>&1

"$VENV/bin/python" - "$CUROBO_OLD" <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import sys
import torch

assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
dist = md.distribution("nvidia-curobo")
direct = json.loads((Path(dist._path) / "direct_url.json").read_text())
assert direct["vcs_info"]["commit_id"] == sys.argv[1], direct
print({"torch": torch.__version__, "curobo_before": direct["vcs_info"]["commit_id"]})
PY

printf '%s\n' '=== PIN ONLY CUROBO; KEEP OFFICIAL DEPENDENCY SET ==='
timeout --signal=INT --kill-after=120s 1800s \
  uv pip install --python "$VENV/bin/python" --reinstall --no-deps \
    "git+https://github.com/NVlabs/curobo.git@$CUROBO_REV" \
    --no-build-isolation

"$VENV/bin/python" - "$CUROBO_REV" <<'PY'
from __future__ import annotations

import importlib.metadata as md
import json
from pathlib import Path
import sys
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState

assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
x = torch.ones(8, device="cuda")
assert float((x * x).sum()) == 8.0
dist = md.distribution("nvidia-curobo")
direct = json.loads((Path(dist._path) / "direct_url.json").read_text())
assert direct["vcs_info"]["commit_id"] == sys.argv[1], direct
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "curobo_version": dist.version,
    "curobo_commit": direct["vcs_info"]["commit_id"],
    "TensorDeviceType": str(TensorDeviceType),
    "Pose": str(Pose),
    "JointState": str(JointState),
})
PY

printf '%s\n' 'R1_CUROBO_V078_NO_DEPS_PIN_OK'
