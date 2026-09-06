#!/usr/bin/env bash
set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
venv=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
run=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v2
date -Is
git -C "$root" status --short
sed -n '450,530p' "$root/rlinf/workers/actor/fsdp_dagger_policy_worker.py"
sed -n '1,180p' "$venv/lib/python3.11/site-packages/openpi/models_pytorch/preprocessing_pytorch.py"
grep -RnE 'precision_processor|cast_forward_inputs|cast_root_forward_inputs|MixedPrecision' "$root/rlinf/models/embodiment/openpi" "$root/rlinf/hybrid_engines/fsdp" --include='*.py'
grep -nE 'success|bc/|Error|Traceback|RuntimeError|rollout|training' "$run/driver.log" | tail -n 35
find "$run/success_data" -type f -printf '%p %s bytes\n'
"$venv/bin/python" - "$run" <<'PY'
import sys,csv,torch
from pathlib import Path
p=Path(sys.argv[1]); rows=list(csv.DictReader((p/'resource.csv').open()))
print('resource columns',rows[0].keys())
for f in (p/'success_data').rglob('batch_*.pt'):
    eps=torch.load(f,map_location='cpu',weights_only=True)
    print('ARCHIVE',f.name,'episodes',len(eps),'queries',sum(map(len,eps)))
    print({k:(tuple(v.shape),str(v.dtype)) for k,v in eps[0][0].items()})
PY
