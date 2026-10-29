import copy,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
assert os.getuid()==1003 and os.uname().nodename=='admin'
T=Path('/home/chenyiteng/deployment-20261030/queue-storage-sz1')
assert '13 passed' in (T/'tests.log').read_text() and not (T/'activation.json').exists()
D=Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67')
R=D/'source';S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,save,same,exact_signal
old=read(D/'identity.json');plan=read(D/'formal-plan.json');prior=plan['predecessor_identity']
assert same(old) and same(prior) and read(D/'training-queue-status.json')['phase']=='PREPARED_WAIT_HANDOFF'
assert not (D/'handoff-complete.json').exists()
rp=read(S/'plan.json');assert all(rp['slots'][g]['external_identity']==old for g in ['6','7'])
save(T/'identity-before.json',old);save(T/'rlt-before.json',rp);save(T/'formal-plan-before.json',plan)
for name in ['queue_storage.py','single_formal_queue.py']:
 dst=R/'tools/pi05_distillation'/name
 if dst.exists():shutil.copy2(dst,T/('before-'+name))
 shutil.copy2(T/'tests'/name,dst)
plan['checkpoint_reserve_gib']={'endpoint':8,'cp_v2':10,'onedp':14,'mfd':14,'fql_fixed_teacher':8}
save(D/'formal-plan.json',plan)
# Only the idle successor coordinator is restarted. Its predecessor still owns 6/7.
exact_signal(old,signal.SIGTERM)
deadline=time.monotonic()+8
while same(old) and time.monotonic()<deadline:time.sleep(.1)
assert not same(old) and same(prior)
(D/'registered.json').rename(T/'registered-before.json')
with (D/'queue.log').open('ab') as log:
 child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_distillation/single_formal_queue.py'),str(D)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
deadline=time.monotonic()+8
while time.monotonic()<deadline:
 me=read(D/'identity.json')
 if me['pid']==child.pid and same(me):break
 assert child.poll() is None;time.sleep(.1)
assert me['pid']==child.pid and same(me)
E=Path('/tmp/chenyiteng-rlt-handoff-20261030')
E.mkdir(mode=0o700,exist_ok=True);assert not E.is_symlink() and E.stat().st_uid==1003
for g in ['6','7']:rp['slots'][g]['external_identity']=me
for g in ['4','5','6','7']:
 slot=rp['slots'][g];identity=slot['external_identity'];assert same(identity)
 permit=E/f'advance-after-owner-exit-g{g}.json'
 save(permit,dict(time=time.time(),identity=identity,authorization='User requests RLT fallback after experiment failure, including disk full',armed=True,requires='exact owner exited AND physical GPU clear; not a completed-training receipt'))
 slot['external_release']=str(permit)
save(S/'plan.json',rp)
save(D/'registered.json',dict(time=time.time(),identity=me,waiting_for=prior,gpus=[6,7],operation='Disk-full safe continuation; scientific configuration unchanged'))
assert same(me) and same(prior) and same(read(S/'owner-identity.json'))
receipt=dict(time=time.time(),old_idle_owner=old,new_idle_owner=me,predecessor_unchanged_alive=True,
             rlt_owner_unchanged_alive=True,reserves_gib=plan['checkpoint_reserve_gib'],
             failure_records_root='/tmp/chenyiteng-queue-'+D.name,
             rlt_permits={g:rp['slots'][g]['external_release'] for g in ['4','5','6','7']},
             healthy_training_not_restarted=True,tests=13)
save(T/'activation.json',receipt);print(json.dumps(receipt))
