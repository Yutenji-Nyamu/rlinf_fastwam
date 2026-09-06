#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"
export PYTHONPATH="$WT:${PYTHONPATH:-}"
git diff --check
git status --short
git diff --stat
"$PY" - <<'PY'
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
cfg = get_openpi_config('pi05_sidney_robotwin')
assert cfg.model.action_horizon == 50
assert cfg.model.pi05 is True
assert cfg.data.extra_delta_transform is False
assert cfg.data.adapt_to_pi is False
assert cfg.data.use_quantile_norm is False
assert cfg.data.assets.asset_id == 'physical-intelligence/robotwin'
print('registry_ok H50 absolute14 mean_std asset_id=physical-intelligence/robotwin')
PY
