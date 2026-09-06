#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
cd "$WT"
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export PYTHONPATH="$WT:${PYTHONPATH:-}"
"$PY" - <<'PY'
import os
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

def flat(value, prefix=''):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items(): out.update(flat(child, f'{prefix}.{key}' if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value): out.update(flat(child, f'{prefix}[{index}]'))
    else: out[prefix] = value
    return out

with initialize_config_dir(version_base=None, config_dir=os.path.join(os.environ['EMBODIED_PATH'], 'config')):
    control = OmegaConf.to_container(compose(config_name='robotwin_adjust_bottle_grpo_openpi_pi05'), resolve=True)
    sidney = OmegaConf.to_container(compose(config_name='robotwin_move_stapler_pad_grpo_openpi_pi05_sidney'), resolve=True)
a, b = flat(control), flat(sidney)
diffs = [(key, a.get(key, '<missing>'), b.get(key, '<missing>')) for key in sorted(a.keys() | b.keys()) if a.get(key, '<missing>') != b.get(key, '<missing>')]
for row in diffs: print(row)
print('resolved_diff_count', len(diffs))
PY
