#!/usr/bin/env bash
set -eu
id
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc
git -C "$root" rev-parse HEAD
git -C "$root" status --short
/usr/bin/python3 - <<'PY'
from pathlib import Path
import re
r=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc')
p=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v4')
lines=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(p/'driver.log').read_text()).splitlines()
print('FAILURE_STACK')
for i,line in enumerate(lines):
    if 'AssertionError' in line:
        for l in lines[max(0,i-48):i+2]: print(l[:600])
        break
for rel,terms in [
 ('rlinf/hybrid_engines/fsdp/strategy/fsdp.py',['def wrap_model','def offload_param','def onload_param']),
 ('rlinf/hybrid_engines/fsdp/strategy/base.py',['def get_model_state_dict','def save_checkpoint','def load_checkpoint']),
 ('rlinf/hybrid_engines/fsdp/strategy/checkpoint.py',['local_shard','def save_checkpoint','def load_checkpoint']),
 ('rlinf/workers/actor/embodied_fsdp_actor_worker.py',['get_rollout_state_dict','def sync_model_to_rollout']),
]:
 path=r/rel
 print('\nFILE',rel,'EXISTS',path.exists())
 if not path.exists(): continue
 ls=path.read_text().splitlines(); shown=set()
 for i,l in enumerate(ls):
  if any(t in l for t in terms):
   for j in range(max(0,i-4),min(len(ls),i+50)):
    if j not in shown: print(f'{j+1}: {ls[j]}'); shown.add(j)
print('RELATED_FILES')
for d in ['rlinf/hybrid_engines/fsdp','rlinf/workers/actor']:
 print('\n'.join(str(x.relative_to(r)) for x in (r/d).rglob('*.py') if 'strategy' in str(x) or 'fsdp_actor' in str(x)))
PY
