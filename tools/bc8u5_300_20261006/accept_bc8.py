import importlib.util,json,os,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261006/ugrow-bc8u5-300-v1');D=S/'bc'
def read(p):return json.loads(Path(p).read_text()) if Path(p).is_file() else None
p=read(D/'plan.json');sp=importlib.util.spec_from_file_location('rt',p['ops']);rt=importlib.util.module_from_spec(sp);sp.loader.exec_module(rt);op,b=rt.configure(D);op.checked()
state=read(D/'status.json');cut=read(S/'cutover-status.json');out={'time':time.time(),'head':p['head'],'source_pins_verified':True,'state':state['stage'],'owner':state['owner'],'owner_alive':b.same(state['owner']),'heartbeat_age':time.time()-(D/'status.json').stat().st_mtime,'error':state.get('error'),'cutover_phase':cut['phase'],'protected_567_alive':all(b.same(q) for q in cut['protected']),'smoke_passed':read(D/'smoke-passed.json'),'formal_first_round':read(D/'formal-first-round.json'),'runs':{},'gpus':b.gpu_snapshot(),'source_published':read(S/'source-published.json')}
for key,row in p['runs'].items():
 r=Path(row['run']);identity=read(r/'runtime/driver-identity.json');m=rt.scalars(r);log=r/'runtime/driver.log'
 rowout={'path':str(r),'identity':identity,'alive':bool(identity and b.same(identity)),'latest':{k:v[-1] for k,v in m.items() if v and k in ('env/success_once','train/bc/actor_loss','train/actor/grad_norm','train/ugrow/round','train/ugrow/weight_nonunit_fraction','train/replay_buffer/filtered_success_episodes','train/replay_buffer/query_records')},'finished':read(r/'runtime/finished.json')}
 if log.is_file():
  rowout['log_age']=time.time()-log.stat().st_mtime
  with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2000));rowout['tail']=f.read().decode(errors='replace')
 out['runs'][key]=rowout
 if identity and b.same(identity):
  scope=rt.scope_state(op,b,p,row,identity);rowout['outside']=scope['outside'];rowout['worker_envs']=[]
  for q in scope['tree']:
   try:
    env=dict(pair.split('=',1) for pair in Path('/proc',str(q['pid']),'environ').read_bytes().decode().split('\0') if '=' in pair)
    if 'ROBOTWIN_PATH' in env:rowout['worker_envs'].append({'pid':q['pid'],'ROBOTWIN_PATH':env.get('ROBOTWIN_PATH'),'PYTHONPATH':env.get('PYTHONPATH')})
   except (OSError,UnicodeDecodeError):pass
out['disk_gib']={x:os.statvfs(x).f_bavail*os.statvfs(x).f_frsize/1024**3 for x in ('/home','/data')}
(S/'latest-acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
