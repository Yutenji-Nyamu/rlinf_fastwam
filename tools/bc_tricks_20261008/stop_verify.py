import json,os,signal,sys,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=S/'trick-ablation-20261008';sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal,gpus,actors
p=read(T/'plan-before.json');pre=read(T/'prestop.json');assert not (T/'concluded.json').exists();assert all(same(q) for q in pre['protected']);assert not same(pre['owner'])
time.sleep(2);snap=gpus();assert all(not snap[r['gpu']]['processes'] for r in pre['rows']);assert all(same(q) for q in pre['protected'])
for r in pre['rows']:assert not actors(p,r['q']['namespace']),r['q']['namespace']
save(T/'concluded.json',dict(time=time.time(),host=p['host'],reason='USER_CONCLUDED',stopped=[dict(gpu=r['gpu'],driver=r['identity'],run=r['q']['run'],retained_checkpoint=r['retained_checkpoint']) for r in pre['rows']],protected_unchanged=pre['protected'],namespace_clear=True,gpu_clear=True,owner_stopped=True))
print('CONCLUDED',p['host'],[r['gpu'] for r in pre['rows']],'GPU_AND_NAMESPACE_CLEAR; OTHER_TRAINING_PRESERVED')
