"""One-shot exact RLT preemption -> finite paired eval -> fresh RLT runtimes."""
import copy,fcntl,json,os,signal,subprocess,sys,time,traceback,yaml
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261020/c20-sz2-g45-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal,gpus,actors
assert os.getuid()==20001 and (T/'repair-context.json').exists() and not (T/'cutover-started.json').exists()
p=read(T/'prepared.json');R=Path(p['source']);plan=read(S/'plan.json');state=read(S/'status.json');owner=read(S/'owner-identity.json')
assert plan==p['scheduler_plan_before'] and same(owner)
rows=read(T/'rlt-before.json')
for row in rows:
 g=row['gpu'];s=state['slots'][g]
 assert s['phase']=='RLT_RUNNING' and s['identity']['pid']==row['identity']['pid'] and same(row['identity']) and s['request']==row['request_path']
protected=[plan['slots'][g]['external_identity'] for g in ('6','7')];assert all(same(v) for v in protected)
argv=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');assert str(S/'ops/owner.py') in argv
owner_env=dict(x.split('=',1) for x in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0') if '=' in x);cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(T/'cutover-started.json',dict(time=time.time(),owner=owner,rlt=rows,plan_before=plan,state_before=state,authorization='User: move queued C20 rank and sweep to SZ2 GPU4/5; current RLT is preemptible fallback.'))
restart=False
try:
 exact_signal(owner,signal.SIGTERM)
 for _ in range(50):
  if not same(owner):break
  time.sleep(1)
 assert not same(owner),'Owner did not release; leave RLT untouched'
 restart=True;lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(S/'plan.json')==plan
 for row in rows:exact_signal(row['identity'],signal.SIGTERM)
 print('RLT_45_STOP_REQUESTED',flush=True)
 for i in range(180):
  snap=gpus()
  if all(not same(row['identity']) and not snap[int(row['gpu'])]['processes'] for row in rows):
   if all(not actors(plan,row['request']['namespace']) for row in rows):break
  if i%30==0:print('WAIT_RLT_RELEASE',i,flush=True)
  time.sleep(1)
 else:raise RuntimeError('RLT release incomplete; no inference launched')
 updated=copy.deepcopy(plan);status=read(S/'status.json');fallbacks=[]
 for row in rows:
  g=row['gpu'];old=row['request'];oldrt=Path(old['runtime']);release=read(oldrt/'finished.json');assert release['cleanup_error'] is None
  cfg=yaml.safe_load((oldrt/'resolved.yaml').read_text());oldroot=Path(old['run'])/Path(old['run']).name/'checkpoints'
  candidates=[Path(cfg['runner']['resume_dir'])]+list(oldroot.glob('global_step_*'))
  required=['actor/dcp_checkpoint/.metadata','actor/dcp_checkpoint/__0_0.distcp','actor/sac_components/rlt_trainer_state/complete.json','actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt','actor/sac_components/target_model/checkpoint_rank_0.pt','actor/sac_components/replay_buffer/rank_0/metadata.json']
  complete=[cp for cp in candidates if all((cp/f).is_file() and (cp/f).stat().st_size>0 for f in required)]
  assert complete
  cp=max(complete,key=lambda x:int(x.name.split('_')[-1]));step=int(cp.name.split('_')[-1])
  run=Path('/data/chenyiteng/results/rlinf-rlt')/f'rlt-sz2-g{g}-after-c20-1020-v1';rt=run/'runtime';assert not run.exists();rt.mkdir(parents=True)
  q=dict(old,run=str(run),runtime=str(rt),namespace=run.name,checkpoint=str(cp),step=step)
  def remap(v):
   if isinstance(v,dict):return {k:remap(x) for k,x in v.items()}
   if isinstance(v,list):return [remap(x) for x in v]
   if isinstance(v,str):
    if v==old['namespace']:return q['namespace']
    if v==old['run'] or v.startswith(old['run']+'/'):return q['run']+v[len(old['run']):]
   return v
  newcfg=remap(cfg);newcfg['runner']['resume_dir']=str(cp)
  save(rt/'resolved.yaml',newcfg);save(rt/'environment.json',remap(read(oldrt/'environment.json')))
  request=T/f'fallback-g{g}.json';save(request,q)
  updated['slots'][g]['fallback']=str(request);status['slots'][g]={'phase':'WAITING_RLT'}
  fallbacks.append(dict(gpu=g,request=str(request),checkpoint=str(cp),step=step,old_release=release))
 save(T/'rlt-released.json',dict(time=time.time(),gpu_namespace_clear=True,fallbacks=fallbacks))
 save(S/'plan.json',updated);save(S/'status.json',status)
 assert all(same(v) for v in protected)
 assert all(not gpus()[g]['processes'] for g in (4,5))
 with (T/'sequence.log').open('x') as log:
  child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(T/'c20_sequence.py'),str(T)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if (T/'sequence-identity.json').exists():break
  assert child.poll() is None;time.sleep(.1)
 identity=read(T/'sequence-identity.json');assert same(identity)
 for g in ('4','5'):
  updated['slots'][g]['external_identity']=identity;updated['slots'][g]['external_release']=str(T/'released.json');status['slots'][g]={'phase':'WAIT_EXTERNAL'}
 save(S/'plan.json',updated);save(S/'status.json',status)
 assert all(updated['slots'][g]==plan['slots'][g] for g in ('6','7'))
 print('SEQUENCE_V2_LAUNCHED',identity,flush=True)
except BaseException:
 save(T/'cutover-error.json',dict(time=time.time(),error=traceback.format_exc()));raise
finally:
 if restart:
  if 'lock' in locals():lock.close()
  with (T/'scheduler-after.log').open('x') as log:
   child=subprocess.Popen(argv,cwd=cwd,env=owner_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(T/'scheduler-restarted.json',dict(time=time.time(),identity=proc(child.pid)))
save(T/'cutover-complete.json',dict(time=time.time(),gpus=[4,5],fallbacks=fallbacks,protected_67_unchanged=True))
print('CUTOVER_DONE',flush=True)
