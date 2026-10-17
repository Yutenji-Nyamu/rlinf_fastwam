"""One-shot authorized GRPO closeout and finite inference/RLT handoff."""
import os,sys,time,signal,json,subprocess,copy,fcntl,traceback
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261017/pi05-policy-budget-eval-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal,gpus,actors
assert os.getuid()==1003 and (T/'preflight.json').exists() and not (T/'cutover-started.json').exists()
p=read(T/'prepared.json');R=Path(p['source']);O=Path(p['output']);plan=read(S/'plan.json');state=read(S/'status.json');owner=read(S/'owner-identity.json')
assert plan==p['plan_before'] and same(owner)
assert all(state['slots'][g]['phase']=='WAIT_EXTERNAL' for g in ('6','7'))
old=p['old_identity'];assert same(old) or Path(p['old_release']).exists()
protected=[plan['slots'][g]['external_identity'] for g in ('4','5')];assert all(same(v) for v in protected)
argv=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');assert str(S/'ops/owner.py') in argv
owner_env=dict(x.split('=',1) for x in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0') if '=' in x);cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(T/'cutover-started.json',dict(time=time.time(),owner=owner,old_identity=old,plan_before=plan,state_before=state,authorization='User ordered closeout of GRPO tau1 and priority inference on its two cards; RLT remains per-card fallback.'))
restart=False
try:
 exact_signal(owner,signal.SIGTERM)
 for _ in range(50):
  if not same(owner):break
  time.sleep(1)
 assert not same(owner),'Owner did not release; leave training untouched'
 restart=True;lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(S/'plan.json')==plan
 was_live=same(old);exact_signal(old,signal.SIGTERM)
 save(T/'grpo-stop-requested.json',dict(time=time.time(),identity=old,was_live=was_live))
 print('GRPO_TAU1_STOP_REQUESTED',flush=True)
 namespace=Path(p['old_run']).name
 for i in range(900):
  if not same(old) and Path(p['old_release']).exists():
   snap=gpus()
   if all(not snap[g]['processes'] for g in (6,7)) and not actors(plan,namespace):break
  if i%30==0:print('WAIT_GRPO_RELEASE',i,flush=True)
  time.sleep(1)
 else:raise RuntimeError('Old scoped cleanup incomplete; no new GPU process launched')
 released=read(p['old_release']);assert released.get('cleanup_error') is None
 save(T/'grpo-released.json',dict(time=time.time(),receipt=released,namespace_clear=True,gpu_clear=True))
 updated=copy.deepcopy(plan);status=read(S/'status.json')
 for g in (6,7):
  assert all(same(v) for v in protected)
  snap=gpus();assert not snap[g]['processes'] and snap[g]['uuid']==plan['slots'][str(g)]['gpu_uuid']
  with (T/f'lane-g{g}.log').open('x') as log:
   child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_policy_eval/lane.py'),str(g)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',POLICY_EVAL_ROOT=str(O),POLICY_EVAL_TRANSACTION=str(T)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(100):
   if (T/f'lane-g{g}-identity.json').exists():break
   assert child.poll() is None,'Lane exited during startup'
   time.sleep(.1)
  identity=read(T/f'lane-g{g}-identity.json');assert same(identity)
  updated['slots'][str(g)]['external_identity']=identity;updated['slots'][str(g)]['external_release']=str(T/f'lane-g{g}-released.json');status['slots'][str(g)]={'phase':'WAIT_EXTERNAL'}
  save(S/'plan.json',updated);save(S/'status.json',status)
  print('INFERENCE_LANE_LAUNCHED',g,identity,flush=True)
 assert all(updated['slots'][g]['fallback']==plan['slots'][g]['fallback'] for g in plan['slots'])
 assert all(updated['slots'][g]==plan['slots'][g] for g in ('4','5'))
except BaseException:
 save(T/'cutover-error.json',dict(time=time.time(),error=traceback.format_exc()));raise
finally:
 if restart:
  if 'lock' in locals():lock.close()
  with (T/'scheduler-after.log').open('x') as log:
   child=subprocess.Popen(argv,cwd=cwd,env=owner_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(T/'scheduler-restarted.json',dict(time=time.time(),identity=proc(child.pid)))
save(T/'cutover-complete.json',dict(time=time.time(),gpus=[6,7],fallbacks_preserved=True,protected_45_unchanged=True))
print('CUTOVER_DONE',flush=True)
