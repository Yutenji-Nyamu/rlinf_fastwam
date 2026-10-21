"""One-shot exact RLT preemption -> finite paired eval -> fresh RLT runtimes."""
import copy,fcntl,json,os,signal,subprocess,sys,time,traceback,yaml
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261021/clean-c20-dense30-sz3-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal,gpus,actors
resume_delayed_owner='--resume-delayed-owner-exit' in sys.argv
resume_released_rlt='--resume-released-rlt' in sys.argv
assert not (resume_delayed_owner and resume_released_rlt)
assert os.getuid()==20001 and (T/'preflight.json').exists() and not (T/'cutover-complete.json').exists()
assert resume_delayed_owner or resume_released_rlt or not (T/'cutover-started.json').exists()
p=read(T/'prepared.json');R=Path(p['source']);O=Path(p['output']);plan=read(S/'plan.json');state=read(S/'status.json');owner=read(S/'owner-identity.json')
assert plan==p['plan_before']
if resume_delayed_owner:
 assert not same(owner) and read(S/'owner-stopped.json')['owner']==owner
 assert 'Owner did not release; leave RLT untouched' in read(T/'cutover-error.json')['error']
 assert read(T/'cutover-started.json')['owner']==owner
else:assert same(owner)
if resume_released_rlt:
 assert "FileNotFoundError" in read(T/'cutover-error.json')['error'] and 'finished.json' in read(T/'cutover-error.json')['error']
rows=p['rlt_before']
for row in rows:
 g=row['gpu'];s=state['slots'][g]
 assert s['identity']['pid']==row['identity']['pid'] and s['request']==row['request_path']
 if resume_released_rlt:assert not same(row['identity']) and s['phase'] in ('RLT_RUNNING','RLT_EXITED')
 else:assert s['phase']=='RLT_RUNNING' and same(row['identity'])
protected=[plan['slots'][g]['external_identity'] for g in ('4','5')];assert all(same(v) for v in protected)
if resume_delayed_owner:
 argv=['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')]
 owner_env=dict(os.environ,CUDA_VISIBLE_DEVICES='');cwd=str(S)
 (T/'cutover-error.json').rename(T/'cutover-owner-delay-attempt1.json')
 save(T/'owner-delay-resumed.json',dict(time=time.time(),owner_stopped=read(S/'owner-stopped.json'),plan_unchanged=True,rlt_still_running=True))
else:
 argv=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');assert str(S/'ops/owner.py') in argv
 owner_env=dict(x.split('=',1) for x in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0') if '=' in x);cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
 if resume_released_rlt:
  (T/'cutover-error.json').rename(T/'cutover-missing-driver-receipt-attempt2.json')
  save(T/'rlt-release-resumed.json',dict(time=time.time(),original_drivers_exited=True,plan_unchanged=True))
 else:save(T/'cutover-started.json',dict(time=time.time(),owner=owner,rlt=rows,plan_before=plan,state_before=state,authorization='User: Shenzhen3 physical6/7, Clean CP140 weak / U tau3 CP70 strong C20 dense under30 takeover sweep, then original per-card RLT.'))
restart=False
try:
 if not resume_delayed_owner:
  exact_signal(owner,signal.SIGTERM)
  for _ in range(180):
   if not same(owner):break
   time.sleep(1)
 assert not same(owner),'Owner did not release; leave RLT untouched'
 restart=True;lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(S/'plan.json')==plan
 for row in rows:
  if same(row['identity']):exact_signal(row['identity'],signal.SIGTERM)
 print('RLT_67_STOP_REQUESTED',flush=True)
 for i in range(180):
  snap=gpus()
  if all(not same(row['identity']) and not snap[int(row['gpu'])]['processes'] for row in rows):
   if all(not actors(plan,row['request']['namespace']) for row in rows):break
  if i%30==0:print('WAIT_RLT_RELEASE',i,flush=True)
  time.sleep(1)
 else:raise RuntimeError('RLT release incomplete; no inference launched')
 updated=copy.deepcopy(plan);status=read(S/'status.json');fallbacks=[]
 for row in rows:
  g=row['gpu'];old=row['request'];oldrt=Path(old['runtime'])
  if (oldrt/'finished.json').exists():
   release=read(oldrt/'finished.json');assert release['cleanup_error'] is None
  else:
   # The observed exit is sufficient only after the exact process, GPU and Ray
   # namespace checks above; do not fabricate a native driver finish receipt.
   release=dict(time=time.time(),kind='externally_observed_release',driver_identity=row['identity'],driver_finish_receipt_present=False,driver_exited=True,gpu_clear=True,namespace_clear=True)
  cfg=yaml.safe_load((oldrt/'resolved.yaml').read_text());oldroot=Path(old['run'])/Path(old['run']).name/'checkpoints'
  candidates=[Path(cfg['runner']['resume_dir'])]+list(oldroot.glob('global_step_*'))
  required=['actor/dcp_checkpoint/.metadata','actor/dcp_checkpoint/__0_0.distcp','actor/sac_components/rlt_trainer_state/complete.json','actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt','actor/sac_components/target_model/checkpoint_rank_0.pt','actor/sac_components/replay_buffer/rank_0/metadata.json']
  complete=[cp for cp in candidates if all((cp/f).is_file() and (cp/f).stat().st_size>0 for f in required)]
  assert complete
  cp=max(complete,key=lambda x:int(x.name.split('_')[-1]));step=int(cp.name.split('_')[-1])
  run=Path('/data/chenyiteng/results/rlinf-rlt')/f'rlt-sz3-g{g}-after-clean-c20-dense30-1021';rt=run/'runtime';assert not run.exists();rt.mkdir(parents=True)
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
 for g in (6,7):
  assert all(same(v) for v in protected)
  snap=gpus();assert not snap[g]['processes'] and snap[g]['uuid']==plan['slots'][str(g)]['gpu_uuid']
  with (T/f'lane-g{g}.log').open('x') as log:
   child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_policy_eval/lane.py'),str(g)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',POLICY_EVAL_ROOT=str(O),POLICY_EVAL_TRANSACTION=str(T),POLICY_EVAL_UID='20001'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(100):
   if (T/f'lane-g{g}-identity.json').exists():break
   assert child.poll() is None,'Lane exited during startup'
   time.sleep(.1)
  identity=read(T/f'lane-g{g}-identity.json');assert same(identity)
  updated['slots'][str(g)]['external_identity']=identity;updated['slots'][str(g)]['external_release']=str(T/f'lane-g{g}-released.json');status['slots'][str(g)]={'phase':'WAIT_EXTERNAL'}
  save(S/'plan.json',updated);save(S/'status.json',status)
  print('INFERENCE_LANE_LAUNCHED',g,identity,flush=True)
 assert all(updated['slots'][g]==plan['slots'][g] for g in ('4','5'))
except BaseException:
 save(T/'cutover-error.json',dict(time=time.time(),error=traceback.format_exc()));raise
finally:
 if restart:
  if 'lock' in locals():lock.close()
  with (T/'scheduler-after.log').open('a') as log:
   child=subprocess.Popen(argv,cwd=cwd,env=owner_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(T/'scheduler-restarted.json',dict(time=time.time(),identity=proc(child.pid)))
save(T/'cutover-complete.json',dict(time=time.time(),gpus=[6,7],fallbacks=fallbacks,protected_45_unchanged=True))
print('CUTOVER_DONE',flush=True)
