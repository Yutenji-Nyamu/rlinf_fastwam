#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"
echo '=== pi0_5 model yaml ==='
sed -n '1,180p' examples/embodiment/config/model/pi0_5.yaml
echo '=== robotwin task yaml ==='
find examples -type f | grep -E 'move_stapler|pi05.*robotwin|robotwin.*pi05' | sort | head -30
echo '=== data config definitions ==='
"$PY" - <<'PY'
import dataclasses, inspect
from openpi.training import config
for cls in [config.DataConfig, config.DataConfigFactory, config.AssetsConfig]:
    print('\n', cls)
    if dataclasses.is_dataclass(cls):
        print([(f.name, f.default) for f in dataclasses.fields(cls)])
    print(inspect.getsource(cls))
PY
echo '=== existing norm stats paths ==='
find /data/chenyiteng -type f -name norm_stats.json 2>/dev/null | grep -Ei 'pi0.?5|robotwin' | head -20 || true
echo '=== sidney processor keys ==='
"$PY" - <<'PY'
from safetensors import safe_open
from pathlib import Path
p=Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab/policy_preprocessor_step_3_normalizer_processor.safetensors')
with safe_open(str(p), framework='pt', device='cpu') as f:
    for k in f.keys():
        print(k, tuple(f.get_tensor(k).shape), f.get_tensor(k).dtype)
PY
