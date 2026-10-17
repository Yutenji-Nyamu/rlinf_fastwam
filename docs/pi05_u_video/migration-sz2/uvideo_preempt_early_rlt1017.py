import json,os,sys,signal,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');D=Path('/data/chenyiteng/deployment-20261017/pi05-u-video-sz2-v2');sys.path.insert(0,str(S/'ops'));from common import read,save,same,exact_signal,gpus,actors
hold=read(D/'owner-held-for-plan-restore.json');plan=read(S/'plan.json');owner=hold['owner'];assert same(owner)
assert not(D/'rlt-preempt-requested.json').exists()
rows=[]
for g in ['4','5']:
 s=hold['status']['slots'][g];assert s['phase']=='RLT_RUNNING' and s['request']==plan['slots'][g]['fallback'];q=read(s['request']);rows.append(dict(gpu=g,status=s,request=q))
save(D/'rlt-preempt-requested.json',dict(time=time.time(),rows=rows,owner=owner,reason='U collection has priority; restore accidentally archived lane plans'))
exact_signal(owner,signal.SIGTERM);exact_signal(owner,signal.SIGCONT)
for _ in range(50):
 if not same(owner):break
 time.sleep(1)
assert not same(owner)
for r in rows:exact_signal(r['status']['identity'],signal.SIGTERM)
for _ in range(180):
 snap=gpus()
 if all(not same(r['status']['identity'])and not snap[int(r['gpu'])]['processes']for r in rows):break
 time.sleep(1)
else:raise RuntimeError('RLT release incomplete')
for r in rows:
 q=r['request'];assert not actors(plan,q['namespace']);p=Path(q['runtime'])/'finished.json';r['release']=read(p)if p.exists()else None
save(D/'rlt-preempt-complete.json',dict(time=time.time(),rows=rows,gpu_namespace_clear=True));print(json.dumps(read(D/'rlt-preempt-complete.json')))
