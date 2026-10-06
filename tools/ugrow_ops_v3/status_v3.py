"""Read-only verification of GPU5 formal and the untouched GPU4 BC owner."""
import hashlib,importlib.util,json,math,os,subprocess,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-rlt-g5-formal-v3')
PREV=S.with_name('ugrow-bc-rlt-g45-v2')
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def tail(p):
    if not p.is_file():return ''
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4500));return f.read().decode(errors='replace')
out={'time':time.time(),'uid':os.getuid(),'lanes':{}}
assert out['uid']==1003
for lane,d in (('bc',PREV/'bc'),('rlt',S/'rlt')):
    p=read(d/'plan.json');spec=importlib.util.spec_from_file_location('rt_'+lane,p['ops']);rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
    op,b=rt.configure(d);op.checked()
    row=p['runs']['formal'];r=Path(row['run']);runtime=r/'runtime';ident=read(runtime/'driver-identity.json');owner=read(d/'owner-identity.json')
    status=read(d/'status.json');scope=rt.scope_state(op,b,p,row,ident);metrics=rt.scalars(r)
    selected={k:rows[-1] for k,rows in metrics.items() if rows and (k=='env/success_once' or k.startswith('train/'))}
    out['lanes'][lane]={'stage':str(d),'run':row,'owner':owner,'owner_alive':b.same(owner) if owner else False,
       'owner_stage':status['stage'] if status else None,'error':status.get('error') if status else None,
       'heartbeat_age':time.time()-(d/'status.json').stat().st_mtime if status else None,
       'driver':ident,'driver_alive':b.same(ident) if ident else False,
       'first_round':read(d/'formal-first-round.json'),'terminal':read(d/'terminal.json'),'return_status':read(d/'return-status.json'),
       'source_head':p['head'],'source_and_pins_verified':True,'config_sha256':hashlib.sha256((runtime/'resolved.yaml').read_bytes()).hexdigest(),
       'resolved_yaml':(runtime/'resolved.yaml').read_text(),'metrics':selected,
       'all_train_metrics_finite':all(math.isfinite(v['value']) for k,rows in metrics.items() if k.startswith('train/') for v in rows),
       'scope':{'gpu':scope['gpu'],'outside':scope['outside'],'processes':scope['tree'],
          'actors':[{k:a.get(k) for k in ('pid','job_id','ray_namespace','name','state')} for a in scope['actors']]},
       'driver_log_age':time.time()-(runtime/'driver.log').stat().st_mtime if (runtime/'driver.log').exists() else None,
       'driver_tail':tail(runtime/'driver.log'),'owner_tail':tail(d/'owner.log')}
d=S/'rlt';ld=d/'rlt-lease';lease=read(ld/'lease.json');stopped=read(ld/'stop-attempt.json');release=read(ld/'rlt-released.json')
out['authorization']=read(d/'formal-authorization-checked.json');out['config_diff']=read(d/'formal-config-diff.json')
out['lease']={'old_identity':stopped['identity'],'signal_set':stopped['signal_set'],'old_identity_alive':b.same(stopped['identity']),
    'released':release['ready'],'release_evidence':release['release_evidence'],'released_gpu':release['gpu'],
    'checkpoint':release['checkpoint'],'return_run':lease['new_run'],'return_dispatched':read(ld/'return-dispatched.json')}
out['gpus']=b.gpu_snapshot()
out['disk_free_gib']={path:os.statvfs(path).f_bavail*os.statvfs(path).f_frsize/1024**3 for path in ('/home','/data')}
print(json.dumps(out))
