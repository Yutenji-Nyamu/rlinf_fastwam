#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
sed -n '50,145p' "$root/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
sed -n '135,175p' "$root/rlinf/hybrid_engines/fsdp/strategy/fsdp.py"
git -C "$root" show dc9b87cc:examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml
git -C "$root" show dc9b87cc:examples/embodiment/config/model/pi0.yaml
sed -n '116,240p' "$root/rlinf/models/embodiment/openpi/openpi_action_model.py"
sed -n '1,115p' "$root/rlinf/models/embodiment/openpi/__init__.py"
sed -n '70,165p' "$venv/lib/python3.11/site-packages/openpi/models_pytorch/pi0_pytorch.py"
PYTHONDONTWRITEBYTECODE=1 "$venv/bin/python" - <<'PY'
import inspect
from torch.distributed.fsdp import MixedPrecision
print('FSDP signature:',inspect.signature(MixedPrecision))
PY
