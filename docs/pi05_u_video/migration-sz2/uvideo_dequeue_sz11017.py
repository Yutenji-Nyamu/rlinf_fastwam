import os,json,sys,time,signal,subprocess,copy,fcntl
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');D=Path('/data/chenyiteng/deployment-20261017/pi05-u-video-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,same,proc,exact_signal
p=read(D/'prepared.json');plan=read(S/'plan.json');owner=read(S/'owner-identity.json');state=read(S/'status.json')
assert os.getuid()==1003 and not (D/'migrated-to-sz2.json').exists() and same(owner)
for g in ['6','7']:
 assert read(D/f'lane-g{g}-status.json')['phase']=='WAIT_PREDECESSOR'
 assert same(plan['slots'][g]['external_identity']) and 'pi05-u-video-v1' in plan['slots'][g]['external_release']
 assert same(p['plan_before']['slots'][g]['external_identity'])
args=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');assert str(S/'ops/owner.py')in args
env=dict(v.split('=',1)for v in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0')if '='in v);cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(D/'migration-started.json',dict(time=time.time(),plan_before=plan,owner_before=owner,target='sz2 physical4/5'))
exact_signal(owner,signal.SIGTERM)
for _ in range(50):
 if not same(owner):break
 time.sleep(1)
assert not same(owner)
lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 updated=copy.deepcopy(plan)
 for g in ['6','7']:
  ident=plan['slots'][g]['external_identity'];exact_signal(ident,signal.SIGTERM)
  for _ in range(30):
   if not same(ident):break
   time.sleep(1)
  assert not same(ident)
  updated['slots'][g]=p['plan_before']['slots'][g];state['slots'][g]={'phase':'WAIT_EXTERNAL'}
 assert all(updated['slots'][g]['fallback']==plan['slots'][g]['fallback']for g in plan['slots'])
 save(S/'plan.json',updated);save(S/'status.json',state)
 save(D/'migrated-to-sz2.json',dict(time=time.time(),target='sz2 physical4/5',not_started=True,old_inference_retained=True,sz1_order='current policy inference -> original per-card RLT'))
finally:
 lock.close()
 with(D/'scheduler-migration.log').open('x')as f:c=subprocess.Popen(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save(D/'scheduler-migration-restarted.json',dict(time=time.time(),identity=proc(c.pid)))
print(json.dumps(read(D/'migrated-to-sz2.json')))
