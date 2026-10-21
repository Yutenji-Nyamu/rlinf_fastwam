"""Restore the authorized SZ3 6/7 dense sweep and corrected RLT fallbacks."""
import copy,fcntl,os,signal,subprocess,sys,time,traceback
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261021/clean-c20-dense30-sz3-v1-recovery1')
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal,gpus,actors
assert os.getuid()==20001 and not(T/'cutover-started.json').exists()
p=read(T/'prepared.json');R=Path(p['source']);O=Path(p['output']);plan=read(S/'plan.json');state=read(S/'status.json');owner=read(S/'owner-identity.json')
assert plan==p['plan_before'] and same(owner)
protected=[plan['slots'][g]['external_identity']for g in ('4','5')]
assert all(same(i)for i in protected)
assert state['slots']['6']['phase']=='RLT_RUNNING' and same(state['slots']['6']['identity'])
assert state['slots']['7']['phase']=='RLT_EXITED' and not same(state['slots']['7']['identity'])
argv=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');assert str(S/'ops/owner.py')in argv
owner_env=dict(v.split('=',1)for v in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0')if '='in v);cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(T/'cutover-started.json',dict(time=time.time(),owner=owner,plan_before=plan,state_before=state))
restart=False
try:
 exact_signal(owner,signal.SIGTERM)
 for _ in range(180):
  if not same(owner):break
  time.sleep(1)
 assert not same(owner),'Owner has not exited; leave its experiments untouched'
 restart=True;lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(S/'plan.json')==plan
 who=state['slots']['6']['identity'];exact_signal(who,signal.SIGTERM);print('RLT_G6_STOP_REQUESTED',flush=True)
 for i in range(180):
  snapshot=gpus()
  if not same(who) and all(not snapshot[g]['processes']for g in (6,7)):
   if all(not actors(plan,read(plan['slots'][str(g)]['fallback'])['namespace'])for g in (6,7)):break
  if i%30==0:print('WAIT_RELEASE',i,flush=True)
  time.sleep(1)
 else:raise RuntimeError('Only stop verified RLT; no sweep until devices and namespace clear')
 save(T/'rlt-released.json',dict(time=time.time(),driver_exited=True,gpu_clear=True,namespace_clear=True,fallbacks=p['fallbacks']))
 updated=copy.deepcopy(plan);status=read(S/'status.json')
 for g in ('6','7'):
  updated['slots'][g]['fallback']=str(T/f'fallback-g{g}.json');status['slots'][g]={'phase':'WAITING_RLT'}
 save(S/'plan.json',updated);save(S/'status.json',status)
 for g in (6,7):
  assert all(same(i)for i in protected)
  snapshot=gpus();assert not snapshot[g]['processes']and snapshot[g]['uuid']==plan['slots'][str(g)]['gpu_uuid']
  with(T/f'lane-g{g}.log').open('x')as log:
   child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_policy_eval/lane.py'),str(g)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',POLICY_EVAL_ROOT=str(O),POLICY_EVAL_TRANSACTION=str(T),POLICY_EVAL_UID='20001'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(200):
   if(T/f'lane-g{g}-identity.json').exists():break
   assert child.poll()is None,'Lane exited during startup'
   time.sleep(.1)
  identity=read(T/f'lane-g{g}-identity.json');assert same(identity)
  updated['slots'][str(g)]['external_identity']=identity;updated['slots'][str(g)]['external_release']=str(T/f'lane-g{g}-released.json');status['slots'][str(g)]={'phase':'WAIT_EXTERNAL'}
  save(S/'plan.json',updated);save(S/'status.json',status);print('LANE_STARTED',g,identity,flush=True)
 assert all(updated['slots'][g]==plan['slots'][g]for g in ('4','5'))
 save(T/'cutover-complete.json',dict(time=time.time(),gpus=[6,7],fallbacks=p['fallbacks'],unrelated_45_unchanged=True))
except BaseException:
 save(T/'cutover-error.json',dict(time=time.time(),error=traceback.format_exc()));raise
finally:
 if restart:
  if 'lock'in locals():lock.close()
  with(T/'scheduler-after.log').open('a')as log:
   child=subprocess.Popen(argv,cwd=cwd,env=owner_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(T/'scheduler-restarted.json',dict(time=time.time(),identity=proc(child.pid)))
print('RECOVERY_CUTOVER_COMPLETE',flush=True)
