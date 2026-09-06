#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import inspect
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
for fn in [get_fsdp_wrap_policy, PI0Pytorch.forward, PaliGemmaWithExpertModel.forward]:
 print('SOURCE',inspect.getfile(fn)); print(inspect.getsource(fn))
PY
