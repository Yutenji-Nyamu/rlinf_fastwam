#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import subprocess,hashlib,json,datetime
wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
print('CST',datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
for name in ('fastwam-current-grpo','pi0-dvac-grpo-current','pi05-robotwin-rl','sidney-pi05-current-rlinf'):
 root=wt/name;print('WORKTREE',root)
 for args in (['rev-parse','HEAD'],['status','--short']):print(subprocess.run(['git','--no-optional-locks','-C',str(root),*args],capture_output=True,text=True).stdout)
 files=['rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py','rlinf/workers/actor/fsdp_actor_worker.py','rlinf/envs/robotwin/robotwin_env.py','rlinf/utils/utils.py']
 if 'fastwam' in name:files += ['rlinf/models/embodiment/fastwam/'+f for f in ('fastwam_policy.py','fastwam_rl.py','builder.py','robotwin_adapter.py')]
 else:files += [str(p.relative_to(root)) for p in (root/'rlinf/models/embodiment/openpi').glob('*.py')]
 for rel in files:
  p=root/rel
  if p.exists():print('SOURCE_JSON',json.dumps({'worktree':name,'rel':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}))
PY
