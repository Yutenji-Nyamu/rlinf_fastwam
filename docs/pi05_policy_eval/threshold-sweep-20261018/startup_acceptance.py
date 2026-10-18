import json,sys,time,os
from pathlib import Path
import numpy as np
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=Path('/data/chenyiteng/deployment-20261018/cp180-threshold-sweep-sz3-v1')
sys.path.insert(0,str(S/'ops'));from common import read,same,gpus
out={'time':time.time(),'owner_alive':same(read(S/'owner-identity.json')),'gpus':gpus(),'rows':[]}
for g in(6,7):
 state=read(T/f'lane-g{g}-status.json');c=read(state['config']);b=Path(c['output']);progress=read(b/'progress.json')if(b/'progress.json').exists()else None
 r=dict(gpu=g,alive=same(read(T/f'lane-g{g}-identity.json')),progress=progress,condition=c['condition'],weight=read(b/'strong-loaded.json'))
 seeds=read(b/'seeds.json');r['seed_count']=len(seeds);r['native_retries']=sum(x['native_retry']for x in seeds)
 r['queries']=[]
 if progress and progress['query']>=1:
  q=np.load(b/'query_000.npz');gate=read(c['threshold_path'])
  expected=(q['u'].mean(axis=1)>gate['high'])&q['active'];assert np.array_equal(q['selected'],expected)
  assert np.all(q['after']<=200) and np.all(q['offered']==50)
  r['first_query_gate_verified']=True;r['queries']=dict(selected=int(q['selected'].sum()),active=int(q['active'].sum()),minimum_actions=int(q['after'].min()),maximum_actions=int(q['after'].max()))
 out['rows'].append(r)
out['accepted']=out['owner_alive']and all(r['alive']and r.get('first_query_gate_verified')and r['seed_count']==32 and r['native_retries']==0 for r in out['rows'])
if out['accepted']:
 for g in(6,7):
  state=read(T/f'lane-g{g}-status.json');pid=state['child']['pid']
  bindings=[int(k)for k,v in out['gpus'].items()if any(p['pid']==pid for p in v['processes'])]
  assert bindings==[g]
 (T/'startup-accepted.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
