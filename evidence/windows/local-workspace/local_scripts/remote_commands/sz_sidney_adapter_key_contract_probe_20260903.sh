set -eu

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
SIDNEY=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

export PYTHONPATH="$WT:${PYTHONPATH:-}"
"$PY" - <<'PY'
from collections import Counter
from pathlib import Path
from safetensors import safe_open

path = Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab/model.safetensors')
with safe_open(path, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    print('source_count', len(keys))
    print('all_model_prefix', all(k.startswith('model.') for k in keys))
    print('top_after_model', Counter(k[len('model.'):].split('.', 1)[0] for k in keys))
    for k in keys:
        if any(token in k for token in ('embed', 'lm_head', 'time_mlp', 'action_in', 'action_out', 'state_proj')):
            print(k, tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype())
PY

echo '=== current target model state contract ==='
"$PY" - <<'PY'
from collections import Counter
import gc
from openpi.models import pi0_config
from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config, OpenPi0ForRLActionPrediction

base = pi0_config.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=True)
cfg = OpenPi0Config(**base.__dict__)
model = OpenPi0ForRLActionPrediction(cfg)
sd = model.state_dict()
print('target_count', len(sd))
print('top', Counter(k.split('.', 1)[0] for k in sd))
for k, v in sd.items():
    if any(token in k for token in ('embed', 'time_mlp', 'action_in', 'action_out', 'state_proj')):
        print(k, tuple(v.shape), v.dtype)
del model, sd
gc.collect()
PY
