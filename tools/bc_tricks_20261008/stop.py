import json,os,signal,sys,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=S/'trick-ablation-20261008';sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal,gpus,actors
p=read(T/'plan-before.json');pre=read(T/'prestop.json');assert not (T/'concluded.json').exists();assert all(same(q) for q in pre['protected']);assert same(pre['owner'])
assert read(S/'owner-identity.json')==pre['owner'];assert read(S/'plan.json')==p
save(T/'switch-hold.json',dict(time=time.time(),reason='USER_CONCLUDED_8_BC; replacement BC has priority',cards=[r['gpu'] for r in pre['rows']]))
if (S/'fallback-enabled.json').exists():(S/'fallback-enabled.json').rename(T/'fallback-enabled-before.json')
exact_signal(pre['owner'],signal.SIGTERM)
for _ in range(50):
 if not same(pre['owner']):break
 time.sleep(1)
assert not same(pre['owner']),'CPU owner still running'
for r in pre['rows']:exact_signal(r['identity'],signal.SIGTERM)
for _ in range(55):
 if not any(same(r['identity']) for r in pre['rows']):break
 time.sleep(1)
for r in pre['rows']:
 if same(r['identity']):exact_signal(r['identity'],signal.SIGKILL)
ns={r['q']['namespace'] for r in pre['rows']};left=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 a=proc(d.name)
 if not a or a['uid']!=os.getuid():continue
 try:e=dict(x.split('=',1) for x in (d/'environ').read_bytes().decode(errors='replace').split('\0') if '='in x)
 except OSError:continue
 if e.get('CLUSTER_NAMESPACE') in ns:left.append(a)
for a in left:exact_signal(a,signal.SIGTERM)
if left:time.sleep(3)
for a in left:exact_signal(a,signal.SIGKILL)
time.sleep(2);snap=gpus();assert all(not snap[r['gpu']]['processes'] for r in pre['rows']);assert all(same(q) for q in pre['protected'])
for r in pre['rows']:assert not actors(p,r['q']['namespace']),r['q']['namespace']
save(T/'concluded.json',dict(time=time.time(),host=p['host'],reason='USER_CONCLUDED',stopped=[dict(gpu=r['gpu'],driver=r['identity'],run=r['q']['run'],retained_checkpoint=r['retained_checkpoint']) for r in pre['rows']],protected_unchanged=pre['protected'],namespace_clear=True,gpu_clear=True,owner_stopped=True))
print('CONCLUDED',p['host'],[r['gpu'] for r in pre['rows']],'GPU_AND_NAMESPACE_CLEAR; OTHER_TRAINING_PRESERVED')
