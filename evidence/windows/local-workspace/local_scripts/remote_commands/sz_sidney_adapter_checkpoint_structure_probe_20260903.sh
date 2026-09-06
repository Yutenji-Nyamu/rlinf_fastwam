set -eu

WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
SIDNEY=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python

echo '=== openpi package files ==='
find "$WT/rlinf/models/embodiment/openpi" -maxdepth 2 -type f -printf '%P\n' | sort | head -n 200

echo '=== checkpoint json ==='
for p in config.json train_config.json policy_preprocessor.json policy_postprocessor.json; do
  echo "--- $p ---"
  "$PY" -m json.tool "$SIDNEY/$p"
done

echo '=== safetensors metadata and keys ==='
"$PY" - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open

root = Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab')
for name in (
    'model.safetensors',
    'policy_preprocessor_step_3_normalizer_processor.safetensors',
    'policy_postprocessor_step_0_unnormalizer_processor.safetensors',
):
    print(f'--- {name} ---')
    with safe_open(root / name, framework='pt', device='cpu') as f:
        keys = list(f.keys())
        print('count', len(keys), 'metadata', f.metadata())
        for key in keys:
            t = f.get_slice(key)
            print(key, tuple(t.get_shape()), t.get_dtype())
PY
