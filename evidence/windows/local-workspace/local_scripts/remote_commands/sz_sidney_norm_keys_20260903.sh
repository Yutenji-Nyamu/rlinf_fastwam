#!/usr/bin/env bash
set -euo pipefail
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
"$PY" - <<'PY'
from safetensors import safe_open
p='/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab/policy_preprocessor_step_3_normalizer_processor.safetensors'
with safe_open(p, framework='pt', device='cpu') as f:
 for k in f.keys(): print(k, tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype())
PY
