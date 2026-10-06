import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,time,subprocess
from pathlib import Path
sys.modules.setdefault('tensorflow',None)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
control=Path('/data/chenyiteng/deployment-20261006/dsrl-u-c20-lease-v2')
sys.path.insert(0,'/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-pi05-u-sz1-20261006/tools/dsrl_u_20261006/ops')
from lease_common import checked,read,save,sha
p,b,op=checked(control)
state=read(control/'status.json');capture=read(control/'c10-capture.json')
out={'time':time.time(),'status':state,'heartbeat_age':time.time()-state['time'],'owner_alive':b.same(state['owner']),
     'protected_unchanged':all(b.same(i) for i in capture['protected'].values()),'runs':{}}
snapshot=b.gpu_snapshot();out['gpu']=snapshot
for gpu,role in (('6','clean'),('7','u')):
    run=Path('/data/chenyiteng/results/rlinf-dsrl-pi05-u-20261006')/(role+'-c20-formal-200-mb256-v2')
    log=run/'runtime/driver.log'
    tail=log.read_text(errors='replace')[-13000:] if log.exists() else ''
    metrics={}
    if (run/'tensorboard').exists():
        a=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});a.Reload()
        metrics={k:[{'step':v.step,'value':v.value,'time':v.wall_time} for v in a.Scalars(k)[-3:]] for k in a.Tags()['scalars']}
    actors=op.active_actors(p,'dsrl-u-20261006-'+run.name)
    op.validate_scoped_actors(p,actors)
    slot=state['slots'][gpu]
    driver=slot.get('driver')
    roots=({driver['pid']} if driver and b.same(driver) else set())|{a['pid'] for a in actors if a.get('pid')}
    tree=op.process_tree(roots,p['uid'])
    outside=[g for g in snapshot if g['gpu']!=int(gpu) and any(v['pid'] in tree for v in g['processes'])]
    out['runs'][role]={'driver_alive':bool(driver and b.same(driver)),'actors':actors,'outside_gpu':outside,
                       'metrics':metrics,'log_age':time.time()-log.stat().st_mtime if log.exists() else None,
                       'tail':tail,'fatal_tokens_in_tail':[x for x in ('Traceback','CUDA out of memory','ActorDiedError','NCCL error') if x in tail]}
out['disk']=subprocess.run(['df','-h','/home','/data'],text=True,capture_output=True).stdout
out['receipts']=[x.name for x in (control/'receipts').glob('*.json')]
accepted=out['owner_alive'] and out['heartbeat_age']<60 and out['protected_unchanged']
summary={}
for role,v in out['runs'].items():
    resident=v['metrics'].get('train/sac/global_resident_transitions',[])
    fresh=v['metrics'].get('train/sac/global_new_transitions',[])
    accepted=accepted and v['driver_alive'] and not v['fatal_tokens_in_tail'] and not v['outside_gpu']
    accepted=accepted and len(resident)>=2 and resident[-1]['value']>resident[-2]['value']
    summary[role]={'resident':resident,'new_transitions':fresh,'errors':v['fatal_tokens_in_tail'],
                   'outside_gpu':v['outside_gpu'],'actors':v['actors']}
if accepted and not (control/'c20-acceptance.json').exists():
    for gpu,role in (('6','clean'),('7','u')):
        actual=snapshot[int(gpu)]['processes']
        envs={a['pid'] for a in out['runs'][role]['actors'] if a.get('name') in ('Env:0','EnvGroup:0')}
        assert any(r['pid'] in envs and r['type']=='C+G' for r in actual)
    save(control/'c20-acceptance.json',{'time':out['time'],'state':state,'heartbeat_age':out['heartbeat_age'],
         'protected_unchanged':True,'gpu':snapshot,'runs':summary,'disk':out['disk'],
         'config_manifest_sha256':sha(control/'c20-configs.json'),'cpu_check_sha256':sha(control/'c20-cpu-check.json')},True)
out['acceptance_exists']=(control/'c20-acceptance.json').exists()
print(json.dumps(out))
