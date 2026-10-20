"""Replace only the specified external pair while keeping one RLT owner."""
import os,sys,time,signal,copy,subprocess,fcntl,traceback
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,same,proc,exact_signal
D=Path(sys.argv[1]);p=read(D/'handoff.json');cards=p['gpus']
assert not(D/'enqueued.json').exists()
plan=read(S/'plan.json');state=read(S/'status.json');owner=read(S/'owner-identity.json')
assert same(owner)and plan==p['plan_before']
for g in cards:assert state['slots'][str(g)]['phase']=='WAIT_EXTERNAL'
argv=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0')
assert str(S/'ops/owner.py')in argv
env=dict(v.split('=',1)for v in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0')if '='in v)
cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(D/'enqueue-started.json',dict(time=time.time(),owner=owner,plan=plan))
exact_signal(owner,signal.SIGTERM)
for _ in range(50):
 if not same(owner):break
 time.sleep(1)
assert not same(owner)
lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 with(D/'sequence.log').open('x')as log:
  c=subprocess.Popen(['/usr/bin/python3','-u','-B',str(D/'grpo_sequence.py'),str(D)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if(D/'sequence-identity.json').exists():break
  assert c.poll()is None
  time.sleep(.1)
 identity=read(D/'sequence-identity.json');assert same(identity)
 updated=copy.deepcopy(plan)
 for g in cards:
  updated['slots'][str(g)].update(external_identity=identity,external_release=str(D/'released.json'))
  state['slots'][str(g)]={'phase':'WAIT_EXTERNAL'}
 assert all(updated['slots'][g]['fallback']==plan['slots'][g]['fallback']for g in plan['slots'])
 assert all(updated['slots'][g]==plan['slots'][g]for g in plan['slots']if int(g)not in cards)
 save(S/'plan.json',updated);save(S/'status.json',state)
 save(D/'enqueued.json',dict(time=time.time(),sequence=identity,cards=cards,fallbacks_preserved=True))
finally:
 lock.close()
 with(D/'scheduler-after.log').open('x')as log:
  c=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(D/'scheduler-after.json',dict(time=time.time(),identity=proc(c.pid)))
print('ENQUEUED',cards,flush=True)
