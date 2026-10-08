import json,os,subprocess,sys,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=S/'trick-ablation-20261008';sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,gpus
pre=read(T/'prestop.json');p=read(T/'plan-next.json');assert (T/'concluded.json').exists() and not same(pre['owner']);assert not (T/'launched.json').exists();assert all(same(q) for q in pre['protected']);snap=gpus();oldstate=read(S/'status.json')
for r in pre['rows']:
 g=str(r['gpu']);q=read(p['slots'][g]['priority']);assert not snap[int(g)]['processes'] and not (Path(q['runtime'])/'started.json').exists();oldstate['slots'][g]=dict(phase='PENDING_PRIORITY')
save(S/'plan.json',p);save(S/'status.json',oldstate)
with (T/'owner.log').open('x') as f:child=subprocess.Popen([p['python'],'-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
save(T/'launched.json',dict(time=time.time(),host=p['host'],owner=proc(child.pid),rows=read(T/'prepared.json')['rows'],fallback_held=True,protected=pre['protected']))
print('DISPATCHED',p['host'],'OWNER',child.pid,'CARDS',[r['gpu'] for r in pre['rows']])
