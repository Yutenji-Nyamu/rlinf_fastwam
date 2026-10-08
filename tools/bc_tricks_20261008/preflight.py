import copy,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
import yaml
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');T=S/'trick-ablation-20261008';sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,gpus
p=read(S/'plan.json');st=read(S/'status.json');host=p['host'];uid=os.getuid();assert uid==p['uid'];assert not T.exists()
owner=read(S/'owner-identity.json');assert same(owner);snap=gpus();cards=[4,5,6,7] if host=='sz1' else [6,7];rows=[]
protected=[]
if host=='sz2':protected=[dict(pid=3503441,uid=uid,start=410425820)]
if host=='sz3':protected=[dict(pid=861883,uid=uid,start=732272066),dict(pid=242804,uid=uid,start=732234854)]
assert all(same(q) for q in protected)
for g in cards:
 slot=p['slots'][str(g)];status=st['slots'][str(g)];q=read(slot['priority']);assert q['kind']=='bc' and q['gpu']==g and status['phase']=='PRIORITY_RUNNING' and same(status['identity'])
 rt=Path(q['runtime']);cfg=yaml.safe_load((rt/'resolved.yaml').read_text());dv=cfg['algorithm']['online_bc']['dvac'];assert dv['chunk_dropout']['enabled'] and dv['alpha_schedule']['enabled']
 cps=[];root=Path(q['run']);required=['actor/local_shard_checkpoint/checkpoint_rank_0.pt','actor/online_bc/rank_0/learner.pt','actor/online_bc/rank_0/success_replay.pt','actor/online_bc/rank_0/signal_tau.pt']
 for cp in (root/root.name/'checkpoints').glob('global_step_*'):
  if all((cp/f).is_file() and (cp/f).stat().st_size>0 for f in required):cps.append(cp)
 assert cps;cp=max(cps,key=lambda z:int(z.name.split('_')[-1]))
 for i,card in snap.items():
  for r in card['processes']:
   if i==g:
    a=proc(r['pid']);assert a and a['uid']==uid
    ev=dict(x.split('=',1) for x in (Path('/proc')/str(a['pid'])/'environ').read_bytes().decode(errors='replace').split('\0') if '='in x);assert ev.get('CLUSTER_NAMESPACE')==q['namespace'],(g,a,ev.get('CLUSTER_NAMESPACE'))
 rows.append(dict(gpu=g,request=slot['priority'],q=q,identity=status['identity'],retained_checkpoint=str(cp),source_head=subprocess.check_output(['git','-C',q['repo'],'rev-parse','HEAD'],text=True).strip(),dvac=dv))
T.mkdir();save(T/'plan-before.json',p);save(T/'status-before.json',st);save(T/'prestop.json',dict(time=time.time(),host=host,owner=owner,rows=rows,protected=protected,gpus=snap,boot_id=p['boot_id']))
print(json.dumps(dict(host=host,owner=owner,targets=[dict(gpu=r['gpu'],namespace=r['q']['namespace'],checkpoint=Path(r['retained_checkpoint']).name,driver=r['identity']['pid']) for r in rows],protected=protected)))
