#!/usr/bin/env bash
set -eu
id
export PYTHONDONTWRITEBYTECODE=1
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3/runtime')
lines=(root/'driver.log').read_text(errors='replace').splitlines()
i=next(i for i,l in enumerate(lines) if 'Fatal Python error' in l)
section=lines[i:i+800]
matches=[]
for j,l in enumerate(section):
    if any(k in l for k in ['Current thread','vector_env.py','_base_task.py','camera.py','Fatal Python error','[ERROR','Exception occurred']):
        matches.extend(section[max(0,j-2):min(len(section),j+5)])
print('FAILURE_BOUNDARY_JSON '+json.dumps({'finished':(root/'finished_at.txt').read_text().strip(),'context':list(dict.fromkeys(matches))},ensure_ascii=False))
PY
