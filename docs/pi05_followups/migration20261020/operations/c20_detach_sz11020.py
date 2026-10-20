import sys,os,signal,time,fcntl,copy,subprocess
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal,gpus
OLD=Path('/data/chenyiteng/deployment-20261019/pi05-noise-early-c20-sz1-v2')
D=Path('/data/chenyiteng/deployment-20261020/early-only-sz1-g45-v1')
assert os.getuid()==1003 and not D.exists()
old=read(OLD/'sequence-identity.json');assert same(old)and read(OLD/'status.json')['phase']=='early'
lanes={g:read(OLD/'early'/f'lane-g{g}-identity.json')for g in(4,5)}
assert all(same(v)for v in lanes.values())
assert not any(r['pid']==old['pid']for card in gpus().values()for r in card['processes'])
D.mkdir(parents=True)
save(D/'prepared.json',dict(old_transaction=str(OLD),old_sequence=old,lanes=lanes,target='sz2 4/5 C20 rank and sweep',reason='User-authorized queue split; preserve running early inference children'))
plan=read(S/'plan.json');owner=read(S/'owner-identity.json');state=read(S/'status.json')
assert all(plan['slots'][str(g)]['external_identity']['pid']==old['pid']for g in(4,5))
argv=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0')
env=dict(v.split('=',1)for v in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0')if '='in v)
cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
exact_signal(old,signal.SIGSTOP) # CPU coordinator only; existing lanes keep running.
exact_signal(owner,signal.SIGTERM)
for _ in range(50):
 if not same(owner):break
 time.sleep(1)
assert not same(owner)
lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 with(D/'bridge.log').open('x')as log:
  c=subprocess.Popen(['/usr/bin/python3','-u','-B','/data/chenyiteng/early_only_bridge1020.py',str(D)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  if(D/'sequence-identity.json').exists():break
  assert c.poll()is None;time.sleep(.1)
 identity=read(D/'sequence-identity.json');assert same(identity)
 updated=copy.deepcopy(plan)
 for g in('4','5'):
  updated['slots'][g].update(external_identity=identity,external_release=str(D/'released.json'))
  state['slots'][g]={'phase':'WAIT_EXTERNAL'}
 save(S/'plan.json',updated);save(S/'status.json',state)
 save(D/'enqueued.json',dict(time=time.time(),identity=identity,existing_lanes_preserved=lanes))
 # Bypass old coordinator's child-terminating SIGTERM handler after replacement.
 exact_signal(old,signal.SIGKILL)
 assert all(same(v)for v in lanes.values())
 save(OLD/'c20-migrated.json',dict(time=time.time(),target='sz2 4/5',replacement=str(D),old_cpu_coordinator_terminated=True))
finally:
 lock.close()
 with(D/'scheduler-after.log').open('x')as log:
  c=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(D/'scheduler-after.json',dict(time=time.time(),identity=proc(c.pid)))
print('C20 detached; existing early lanes still running; SZ1 4/5 then RLT.')
