#!/usr/bin/env bash
set -euo pipefail

source /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/activate
python - <<'PY'
from pathlib import Path
import torch
from torch.distributed.tensor import DTensor

p = Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-localshard-v2-envoffload/fastwam_grpo_smoke1_current/checkpoints/global_step_1/actor/local_shard_checkpoint/checkpoint_rank_0.pt')
state = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
print('top', sorted(state))
model = state['model']
for k in list(model)[:12]:
    v = model[k]
    print('model', k, type(v).__name__, tuple(v.shape) if hasattr(v, 'shape') else None, 'dtensor', isinstance(v, DTensor))
opt = state['optimizers']
print('optim type', type(opt).__name__, 'keys', list(opt) if isinstance(opt, dict) else None)
if isinstance(opt, dict):
    ostate = opt.get('state', {})
    print('optim state count', len(ostate))
    for pid, entry in list(ostate.items())[:2]:
        print('optim pid', pid)
        for ek, ev in entry.items():
            print(' optim', ek, type(ev).__name__, tuple(ev.shape) if hasattr(ev, 'shape') else None, 'dtensor', isinstance(ev, DTensor))
PY
