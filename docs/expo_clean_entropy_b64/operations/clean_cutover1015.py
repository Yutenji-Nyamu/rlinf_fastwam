"""One-shot authorized replacement; reuse existing pair and per-card owners."""
import os,json,sys,time,signal,subprocess,copy,fcntl,traceback
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261015/expo-clean-entropy-b64-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal,gpus
assert os.getuid()==1003 and (T/'prepared.json').exists()and not(T/'cutover-started.json').exists()
q=read(T/'request.json');plan=read(S/'plan.json');owner=read(S/'owner-identity.json');state=read(S/'status.json');old=plan['slots']['4']['external_identity'];release=Path(plan['slots']['4']['external_release'])
assert old==plan['slots']['5']['external_identity'] and same(old)and old['pid']==866004 and same(owner)
assert all(same(plan['slots'][g]['external_identity'])for g in ['6','7'])
assert all(state['slots'][g]['phase']=='WAIT_EXTERNAL'for g in ['4','5','6','7'])
argv=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');assert str(S/'ops/owner.py')in argv
env=dict(x.split('=',1)for x in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0')if'='in x);cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(T/'cutover-started.json',dict(time=time.time(),old=old,owner=owner,plan_before=plan,state_before=state,old_release=str(release)))
restart=False
try:
 exact_signal(owner,signal.SIGTERM)
 for _ in range(50):
  if not same(owner):break
  time.sleep(1)
 assert not same(owner),'Scheduler did not release its lock; old training retained'
 restart=True
 lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(S/'plan.json')==plan
 exact_signal(old,signal.SIGTERM);save(T/'old-stop-requested.json',dict(time=time.time(),identity=old));print('OLD_STOP_REQUESTED',flush=True)
 for i in range(900):
  if not same(old)and release.exists():break
  if i%30==0:print('WAIT_OLD_RELEASE',i,flush=True)
  time.sleep(1)
 assert not same(old)and release.exists(),'Old training still stopping; no forced kill or new launch'
 released=read(release);assert released.get('owned_contexts_clear')and released['cleanup']['ok']
 snap=gpus();assert all(not snap[g]['processes']and snap[g]['uuid']==u for g,u in zip(q['gpus'],q['uuids']))
 save(T/'old-released.json',released);print('OLD_RELEASED_GPU45_CLEAR',flush=True)
 assert all(same(plan['slots'][g]['external_identity'])for g in ['6','7'])
 rt=Path(q['runtime']);assert not(rt/'driver.json').exists()
 with(T/'pair-owner.log').open('x')as log:
  child=subprocess.Popen(['/usr/bin/python3','-u',str(Path(q['source'])/'tools/expo_fm_u_20261010/pair_owner.py'),str(T/'request.json')],cwd=q['source'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(200):
  if(rt/'driver.json').exists():break
  assert child.poll()is None,(T/'pair-owner.log').read_text()
  time.sleep(.1)
 ident=read(rt/'driver.json');assert same(ident)
 updated=copy.deepcopy(plan);status=read(S/'status.json')
 for g in ['4','5']:
  updated['slots'][g]['external_identity']=ident;updated['slots'][g]['external_release']=str(rt/'released.json');status['slots'][g]={'phase':'WAIT_EXTERNAL'}
 assert all(updated['slots'][g]==plan['slots'][g]for g in ['6','7'])
 assert all(updated['slots'][g]['fallback']==plan['slots'][g]['fallback']for g in ['4','5'])
 save(S/'plan.json',updated);save(S/'status.json',status);save(T/'launched.json',dict(time=time.time(),identity=ident,pair_owner=proc(child.pid),request=str(T/'request.json'),other_slots_preserved=True,fallbacks_preserved=True));print('NEW_CLEAN_LAUNCHED',ident,flush=True)
except BaseException:
 save(T/'cutover-error.json',dict(time=time.time(),error=traceback.format_exc()));raise
finally:
 if restart:
  if 'lock'in locals():lock.close()
  with(T/'scheduler-after.log').open('x')as log:
   child=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(T/'scheduler-restarted.json',dict(time=time.time(),identity=proc(child.pid)))
print('CUTOVER_DONE',flush=True)
