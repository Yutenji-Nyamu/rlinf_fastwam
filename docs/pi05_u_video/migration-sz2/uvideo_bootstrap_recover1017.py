import os,sys,time,signal,subprocess,copy,fcntl,json
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');D=Path('/data/chenyiteng/deployment-20261017/pi05-u-video-sz2-v2');O=Path('/data/chenyiteng/results/pi05-u-video/20261017-sz2-v2')
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal,gpus
assert (D/'bootstrap-checked.json').exists() and not(D/'bootstrap-recovery-started.json').exists() and not any((O/'batches').iterdir())
assert all(read(p)['returncode']==78 for p in D.glob('*_b*-result.json'))
plan=read(S/'plan.json');owner=read(S/'owner-identity.json');assert same(owner)
assert all(not gpus()[g]['processes']for g in [0,4,5]);protected=[plan['slots'][g]['external_identity']for g in ['6','7']];assert all(same(i)for i in protected)
args=Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');env=dict(v.split('=',1)for v in Path(f"/proc/{owner['pid']}/environ").read_bytes().decode().split('\0')if '='in v);cwd=os.readlink(f"/proc/{owner['pid']}/cwd")
save(D/'bootstrap-recovery-started.json',dict(time=time.time(),plan_before=plan,owner=owner,reason='Initial scope manifest group-write bit rejected before any batch/model started; mode corrected and actual bootstrap passed on both card environments'))
exact_signal(owner,signal.SIGTERM)
for _ in range(50):
 if not same(owner):break
 time.sleep(1)
assert not same(owner);lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 for g in ['4','5']:
  ident=plan['slots'][g]['external_identity'];exact_signal(ident,signal.SIGTERM);exact_signal(ident,signal.SIGCONT)
 for _ in range(30):
  if all(not same(plan['slots'][g]['external_identity'])for g in ['4','5']):break
  time.sleep(1)
 assert all(not same(plan['slots'][g]['external_identity'])for g in ['4','5']) and not any((O/'batches').iterdir())
 A=D/'bootstrap-failed';A.mkdir();moved=[]
 for p in list(D.iterdir()):
  if p.is_file() and (p.name.startswith(('lane-g4','lane-g5')) or p.name.endswith(('-g4.log','-g5.log')) or ('_b'in p.name and p.name.endswith('-result.json'))):
   assert p.stat().st_uid==os.getuid() and p.resolve().parent==D.resolve();p.rename(A/p.name);moved.append(p.name)
 if(O/'summary.json').exists():(O/'summary.json').rename(A/'summary-before.json')
 m=read(O/'manifest.json');m['source_commit']='d73a1b5d1';save(O/'manifest.json',m)
 for p in (O/'configs').glob('*_b?.json'):
  c=read(p);c['source_commit']='d73a1b5d1';save(p,c)
 R=Path(m['source']);updated=copy.deepcopy(plan);status=read(S/'status.json')
 for g in ['4','5']:
  with(D/f'lane-g{g}.log').open('x')as log:c=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_u_video/lane.py'),g],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',U_VIDEO_ROOT=str(O),U_VIDEO_TRANSACTION=str(D)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  for _ in range(100):
   if(D/f'lane-g{g}-identity.json').exists():break
   assert c.poll()is None;time.sleep(.1)
  ident=read(D/f'lane-g{g}-identity.json');assert same(ident);updated['slots'][g]['external_identity']=ident;status['slots'][g]={'phase':'WAIT_EXTERNAL'};save(S/'plan.json',updated);save(S/'status.json',status)
 save(D/'bootstrap-recovery-complete.json',dict(time=time.time(),archived_startup_files=moved,full_original_jobs_retained=True,source_commit='d73a1b5d1',fallbacks_preserved=all(plan['slots'][g]['fallback']==updated['slots'][g]['fallback']for g in plan['slots']),lanes={g:updated['slots'][g]['external_identity']for g in ['4','5']}))
finally:
 lock.close()
 with(D/'scheduler-recovery.log').open('x')as log:c=subprocess.Popen(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(D/'scheduler-recovery-restarted.json',dict(time=time.time(),identity=proc(c.pid)))
print(json.dumps(read(D/'bootstrap-recovery-complete.json')))
