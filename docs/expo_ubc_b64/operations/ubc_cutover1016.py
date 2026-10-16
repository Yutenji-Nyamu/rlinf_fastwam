"""Authorized one-shot cutover using the existing scoped drivers and owners."""
import os,json,sys,time,signal,subprocess,copy,fcntl,traceback
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261016/expo-ubc-b64-tau123-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal,gpus,actors
assert os.getuid()==20001 and (T/'prepared.json').exists() and not (T/'cutover-started.json').exists()
prepared=read(T/'prepared.json');groups=prepared['groups'];plan=read(S/'plan.json');owner=read(S/'owner-identity.json');state=read(S/'status.json')
assert plan==prepared['plan_before'] and same(owner)
assert all(state['slots'][g]['phase']=='WAIT_EXTERNAL' and same(s['external_identity']) for g,s in plan['slots'].items())
for p in groups:
 assert same(p['old_identity']) and not Path(p['old_release']).exists()
 for g in p['gpus']:assert plan['slots'][str(g)]['external_identity']==p['old_identity']
 q=read(p['request']);assert not (Path(q['runtime'])/'driver.json').exists()
argv=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');assert str(S/'ops/owner.py') in argv
env=dict(x.split('=',1) for x in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0') if '=' in x);cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(T/'cutover-started.json',dict(time=time.time(),owner=owner,plan_before=plan,state_before=state,groups=groups))
restart=False
try:
 exact_signal(owner,signal.SIGTERM)
 for _ in range(50):
  if not same(owner):break
  time.sleep(1)
 assert not same(owner),'Scheduler did not release; training retained'
 restart=True;lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(S/'plan.json')==plan
 for p in groups:
  exact_signal(p['old_identity'],signal.SIGTERM)
  save(T/('tau'+str(p['tau'])+'-old-stop-requested.json'),dict(time=time.time(),identity=p['old_identity'],old_request=p['old_request']))
 print('AUTHORIZED_OLD_STOPS_REQUESTED',flush=True)
 for p in groups:
  old=p['old_identity'];release=Path(p['old_release']);oldq=read(p['old_request'])
  for i in range(900):
   if not same(old) and release.exists():
    snap=gpus()
    if all(not snap[g]['processes'] for g in p['gpus']):
     if oldq.get('namespace') and actors(plan,oldq['namespace']):pass
     else:break
   if i%30==0:print('WAIT_OLD_RELEASE',p['tau'],i,flush=True)
   time.sleep(1)
  else:raise RuntimeError('Old scoped cleanup incomplete; no forced kill or new launch')
  released=read(release)
  if oldq.get('namespace'):assert released.get('cleanup_error') is None and not actors(plan,oldq['namespace'])
  else:assert released.get('owned_contexts_clear') and released['cleanup']['ok']
  save(T/('tau'+str(p['tau'])+'-old-released.json'),dict(time=time.time(),identity=old,receipt=released,namespace_clear=True,gpu_clear=True))
  print('OLD_RELEASED',p['tau'],p['gpus'],flush=True)
 updated=copy.deepcopy(plan);status=read(S/'status.json')
 for p in groups:
  q=read(p['request']);rt=Path(q['runtime']);snap=gpus()
  assert all(not snap[g]['processes'] and snap[g]['uuid']==u for g,u in zip(q['gpus'],q['uuids']))
  assert all(same(v['external_identity']) for v in p['protected_slots'].values())
  with (T/('tau'+str(q['tau'])+'-pair-owner.log')).open('x') as log:
   child=subprocess.Popen(['/usr/bin/python3','-u',str(Path(q['source'])/'tools/expo_fm_u_20261010/pair_owner.py'),p['request']],cwd=q['source'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(200):
   if (rt/'driver.json').exists():break
   assert child.poll() is None,'Pair owner exited; inspect log'
   time.sleep(.1)
  ident=read(rt/'driver.json');assert same(ident)
  for g in map(str,q['gpus']):
   updated['slots'][g]['external_identity']=ident;updated['slots'][g]['external_release']=str(rt/'released.json');status['slots'][g]={'phase':'WAIT_EXTERNAL'}
  assert all(updated['slots'][g]['fallback']==plan['slots'][g]['fallback'] for g in plan['slots'])
  save(S/'plan.json',updated);save(S/'status.json',status)
  save(T/('tau'+str(q['tau'])+'-launched.json'),dict(time=time.time(),identity=ident,pair_owner=proc(child.pid),request=p['request'],fallbacks_preserved=True))
  print('NEW_UBC_LAUNCHED',q['tau'],q['gpus'],ident,flush=True)
except BaseException:
 save(T/'cutover-error.json',dict(time=time.time(),error=traceback.format_exc()));raise
finally:
 if restart:
  if 'lock' in locals():lock.close()
  with (T/'scheduler-after.log').open('x') as log:
   child=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(T/'scheduler-restarted.json',dict(time=time.time(),identity=proc(child.pid)))
save(T/'cutover-complete.json',dict(time=time.time(),taus=[p['tau'] for p in groups]))
print('CUTOVER_DONE',flush=True)
