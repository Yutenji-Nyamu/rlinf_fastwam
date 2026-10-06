import fcntl,importlib.util,json,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-budget-400-2000-v1')
out={'time':time.time(),'lanes':{}}
for lane in ('bc','rlt'):
    d=S/lane;p=json.loads((d/'plan.json').read_text());sp=importlib.util.spec_from_file_location('rt_'+lane,p['ops']);rt=importlib.util.module_from_spec(sp);sp.loader.exec_module(rt)
    op,b=rt.configure(d);op.checked();e=rt.read(d/'extension.json')
    def read(p):return rt.read(p) if p.is_file() else None
    owner=read(d/'extension-identity.json');state=read(d/'extension-status.json')
    with (Path(p['lease_dir'])/'operation.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);guard=False
        except BlockingIOError:guard=True
    old=Path(e['original_stage']);oldp=rt.read(old/'plan.json');oldrt=rt.module(oldp['ops'],'old_'+lane);oldop,_=oldrt.configure(old);oldop.checked()
    scope=oldrt.scope_state(oldop,b,oldp,oldp['runs']['formal'],e['original_driver']);m=oldrt.scalars(e['original_run'])
    out['lanes'][lane]={'extension':e,'owner':owner,'owner_alive':bool(owner and b.same(owner)), 'armed':read(d/'armed.json'),
       'state':state,'heartbeat_age':time.time()-(d/'extension-status.json').stat().st_mtime if state else None,'lease_guard_held':guard,
       'original_owner_alive':b.same(e['original_owner']),'original_driver_alive':b.same(e['original_driver']),
       'original_metrics':{k:rows[-1] for k,rows in m.items() if rows and k in ('env/success_once','train/rlt/global_min_replay_size','train/rlt/update_step','train/ugrow/round')},
       'gpu':scope['gpu'],'outside':scope['outside'],'plan_pins_verified':True,'config_diff':read(d/'config-diff.json'),
       'pruned':[read(f) for f in d.glob('prune-step-*.json')],'error_log':(d/'extension.log').read_text()[-1800:] if (d/'extension.log').is_file() else ''}
out['gpus']=b.gpu_snapshot();print(json.dumps(out))
