#!/usr/bin/env bash
set -euo pipefail
WT=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf
PY=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python
SRC=/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab
OUT=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
cd "$WT"
export PYTHONPATH="$WT:${PYTHONPATH:-}"
"$PY" - <<'PY'
import json
from pathlib import Path
import torch
from safetensors import safe_open

src = Path('/data/chenyiteng/models/lerobot/sidney-pi05-robotwin-e49e2ab/model.safetensors')
out_dir = Path('/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab')
dst = out_dir / 'model.safetensors'
with safe_open(src, framework='pt', device='cpu') as a, safe_open(dst, framework='pt', device='cpu') as b:
    assert len(a.keys()) == len(b.keys()) == 813
    for source_key in a.keys():
        target_key = source_key.removeprefix('model.')
        assert target_key in b.keys()
        assert torch.equal(a.get_tensor(source_key), b.get_tensor(target_key)), target_key

manifest_path = out_dir / 'conversion_manifest.json'
manifest = json.loads(manifest_path.read_text())
dtype_mismatches = manifest.pop('dtype_mismatches', [])
manifest['dtype_mismatch_count'] = len(dtype_mismatches)
manifest['dtype_note'] = (
    'Source mixed BF16/FP32 is preserved; a fresh target model is FP32 before '
    'RLinf applies its maintained mixed-precision policy.'
)
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
norm = json.loads((out_dir / 'physical-intelligence/robotwin/norm_stats.json').read_text())['norm_stats']
assert len(norm['state']['mean']) == len(norm['state']['std']) == 14
assert len(norm['actions']['mean']) == len(norm['actions']['std']) == 14
print('tensor_value_parity=813/813')
print('norm_contract=mean_std_state14_action14')
print(json.dumps(manifest, indent=2))
PY
du -sh "$OUT"
stat -c '%n %s' "$OUT/model.safetensors" "$OUT/conversion_manifest.json" "$OUT/physical-intelligence/robotwin/norm_stats.json"
