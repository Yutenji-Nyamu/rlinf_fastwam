set -euo pipefail

echo '=== identity ==='
hostname
id
date '+%F %T %Z'

CKPT=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/lerobot-v060-sidney-py310/bin/python

echo '=== checkpoint files ==='
find "$CKPT" -maxdepth 2 -type f -printf '%P\t%s bytes\n' | sort

echo '=== checkpoint json contracts ==='
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab')
for p in sorted(root.glob('*.json')):
    print(f'--- {p.name}')
    obj = json.loads(p.read_text())
    print(json.dumps(obj, indent=2, sort_keys=True)[:18000])
PY

echo '=== safetensors structure ==='
"$PY" - <<'PY'
from collections import Counter
from pathlib import Path
from safetensors import safe_open

root = Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab')
for p in sorted(root.glob('*.safetensors')):
    print(f'--- {p.name}')
    with safe_open(str(p), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        print('count', len(keys), 'metadata', f.metadata())
        prefix2 = Counter('.'.join(k.split('.')[:2]) for k in keys)
        print('prefix2', dict(prefix2.most_common(30)))
        shown = keys if len(keys) <= 30 else keys[:15] + keys[-15:]
        for k in shown:
            t = f.get_slice(k)
            print(k, tuple(t.get_shape()), t.get_dtype())
PY

echo '=== dedicated worktree state ==='
git -C "$WT" status --short --branch
git -C "$WT" rev-parse HEAD

echo '=== current converter files ==='
sed -n '1,320p' "$WT/rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi_rlinf.py"
echo '--- convert.py'
sed -n '1,320p' "$WT/rlinf/utils/ckpt_convertor/openpi/convert.py"

echo '=== relevant dataconfig and model loader ==='
sed -n '1,260p' "$WT/rlinf/models/embodiment/openpi/dataconfig/__init__.py"
sed -n '1,300p' "$WT/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py"
sed -n '1,240p' "$WT/rlinf/models/embodiment/openpi/__init__.py"

echo '=== current pi05 yaml ==='
sed -n '1,320p' "$WT/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_pi05.yaml"
