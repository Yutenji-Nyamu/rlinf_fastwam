#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
for name in ('fastwam-current-grpo','pi0-dvac-grpo-current','pi05-robotwin-rl','sidney-pi05-current-rlinf'):
 for rel in ('rlinf/workers/actor/embodied_fsdp_actor_worker.py','rlinf/algorithms/utils.py','rlinf/envs/robotwin/seeds/eval_seeds.json'):
  p=wt/name/rel
  if p.exists():print('SOURCE_JSON',json.dumps({'worktree':name,'rel':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text() if p.suffix=='.py' else None}))
 # These commits are checked for lineage only; no fetch or checkout.
 for sha in ('dfef2da6589de9162b6c4eebbc1e2eed9237d669',):
  r=subprocess.run(['git','--no-optional-locks','-C',str(wt/name),'merge-base','--is-ancestor',sha,'HEAD'],capture_output=True,text=True);print('UPSTREAM_ANCESTOR',name,sha,r.returncode,r.stderr[:300])
PY
