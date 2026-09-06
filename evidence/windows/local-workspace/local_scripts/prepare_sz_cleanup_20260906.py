"""Prepare, never execute, the user's exact >1GiB checkpoint-only deletion list."""
import collections,json,re
from pathlib import Path

root=Path(__file__).resolve().parents[1]
doc=root/'docs/server-admin'
x=json.loads((doc/'SZ_ARCHIVE_ALL_RESULTS_INVENTORY_20260906.json').read_text(encoding='utf-8'))
family_prefixes=('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/',
 '/data/chenyiteng/results/rlinf-shenzhen/online-bc/',
 '/data/chenyiteng/results/rlinf-current-dsrl/')
targets=[];protected=[];runs=[];unresolved=[]
for cr,gens in x['checkpoints'].items():
 smoke=bool(re.search(r'(^|[/_-])smokes?([/_-]|\d|$)',cr,re.I))
 if not smoke and not cr.startswith(family_prefixes):continue
 latest=gens[-1];lf={f['path'].split(latest['path']+'/',1)[1]:f for f in latest['files']}
 big=[f for f in latest['files'] if f['bytes']>1024**3]
 # No last state is declared complete merely from a directory name.
 complete=False
 if '/rlinf-current-dsrl/' in cr:
  required=[f'actor/{sub}/checkpoint_rank_{rank}.pt' for sub in ['local_shard_checkpoint','sac_components/target_model'] for rank in [0,1]]
  complete=all(k in lf and lf[k]['bytes']>1024**3 for k in required)
 elif '/online-bc/' in cr:
  required=['actor/local_shard_checkpoint/checkpoint_rank_0.pt','actor/model_state_dict/full_weights.pt',
   'actor/online_bc/rank_0/success_replay.pt','actor/online_bc/rank_0/learner.pt']
  complete=all(k in lf and lf[k]['bytes']>0 for k in required)
 elif '/pi05-sidney/' in cr:
  required=['actor/local_shard_checkpoint/checkpoint_rank_0.pt','actor/local_shard_checkpoint/checkpoint_rank_1.pt','actor/model_state_dict/full_weights.pt']
  complete=all(k in lf and lf[k]['bytes']>1024**3 for k in required)
 if not smoke and not complete:
  unresolved.append({'root':cr,'reason':'latest necessary files not verified; keep all','latest_step':latest['step']});continue
 selected=[]
 for g in gens:
  if not smoke and g['step']==latest['step']:
   protected.extend(g['files']);continue
  for f in g['files']:
   if f['bytes']<=1024**3:continue
   if f['path'] in x['open_files']:
    unresolved.append({'path':f['path'],'reason':'open file','pids':x['open_files'][f['path']]});continue
   assert f['uid']==1003 and f['nlink']==1
   targets.append(dict(f,checkpoint_root=cr,step=g['step'],is_smoke=smoke));selected.append(f)
 if selected:runs.append({'root':cr,'smoke':smoke,'keep_step':None if smoke else latest['step'],
  'files':len(selected),'gib':sum(f['allocated'] for f in selected)/1024**3})

assert len({f['path'] for f in targets})==len(targets)
result={'inventory_time':x['time'],'authorization':'2026-09-06 user: all smoke large checkpoints, non-last Sidney/DSRL/online-BC large checkpoints',
 'threshold_bytes':1024**3,'targets':targets,'protected_files':protected,'runs':runs,'unresolved':unresolved,
 'files':len(targets),'allocated_gib':sum(f['allocated'] for f in targets)/1024**3,
 'note':'Plan only. Executor must refresh stat, latest generation, active references and open descriptors. Never delete a directory.'}
dest=doc/'SZ_CHECKPOINT_CLEANUP_ALLOWLIST_20260906.json'
assert not dest.exists()
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['targets','protected_files']},ensure_ascii=True,indent=2))
