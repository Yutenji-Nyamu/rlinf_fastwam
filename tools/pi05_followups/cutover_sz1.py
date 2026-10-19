"""One authorized EXPO closeout and exact SZ1 GPU4/5 queue replacement."""
import copy,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
D=Path('/data/chenyiteng/deployment-20261019/pi05-noise-early-c20-sz1-v1')
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal
assert os.getuid()==1003 and (D/'prepared.json').exists()and not(D/'cutover-started.json').exists()
p=read(D/'prepared.json');R=Path(p['source']);plan=read(S/'plan.json');state=read(S/'status.json');owner=read(S/'owner-identity.json')
assert plan==p['plan_before'] and same(owner)
old=plan['slots']['4']['external_identity']
assert old==plan['slots']['5']['external_identity']and same(old)
assert all(state['slots'][g]['phase']=='WAIT_EXTERNAL'for g in ('4','5','6','7'))
assert all(same(plan['slots'][g]['external_identity'])for g in ('6','7'))
argv=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0')
assert str(S/'ops/owner.py')in argv
environment=dict(s.split('=',1)for s in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0')if'='in s)
cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(D/'cutover-started.json',{'time':time.time(),'old':old,'owner':owner,'plan_before':plan,
     'authorization':'User: close SZ1 GPU4/5 Clean; noise signal -> early2 sweep -> C20 ranking/sweep -> original RLT.'})
restart=False;lock=None;committed=False;child=None
try:
 exact_signal(owner,signal.SIGTERM)
 for _ in range(50):
  if not same(owner):break
  time.sleep(1)
 assert not same(owner),'CPU scheduler did not exit; training untouched'
 restart=True;lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(S/'plan.json')==plan
 with(D/'sequence.log').open('x')as log:
  child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_followups/sequence.py'),str(D)],
        cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if(D/'status.json').exists():break
  assert child.poll()is None;time.sleep(.1)
 ident=read(D/'sequence-identity.json');assert same(ident)
 assert read(D/'status.json')['phase']=='WAIT_PREDECESSOR'
 updated=copy.deepcopy(plan)
 for g in ('4','5'):
  updated['slots'][g]['external_identity']=ident
  updated['slots'][g]['external_release']=str(D/'released.json')
 assert all(updated['slots'][g]==plan['slots'][g]for g in ('6','7'))
 assert all(updated['slots'][g]['fallback']==plan['slots'][g]['fallback']for g in plan['slots'])
 save(S/'plan.json',updated);save(S/'status.json',state);committed=True
 save(D/'enqueued.json',{'time':time.time(),'identity':ident,'gpus':[4,5],'fallbacks_preserved':True,'other_slots_preserved':True})
except BaseException:
 save(D/'cutover-error.json',{'time':time.time(),'error':traceback.format_exc()})
 if child and child.poll()is None and not committed:exact_signal(proc(child.pid),signal.SIGTERM)
 raise
finally:
 if lock:lock.close()
 if restart:
  with(D/'scheduler-after.log').open('x')as log:
   scheduler=subprocess.Popen(argv,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(D/'scheduler-restarted.json',{'time':time.time(),'identity':proc(scheduler.pid)})
assert committed and same(ident)
exact_signal(old,signal.SIGTERM)
save(D/'old-stop-requested.json',{'time':time.time(),'identity':old,'reason':'User authorized closeout after conclusion; preserve final checkpoint and lightweight plot artifacts.'})
save(D/'cutover-complete.json',{'time':time.time(),'sequence':ident,'old_stop_requested':old,'gpus':[4,5]})
print(json.dumps(read(D/'cutover-complete.json')))
