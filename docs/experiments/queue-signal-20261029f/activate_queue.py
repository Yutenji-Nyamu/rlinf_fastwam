import copy,json,os,signal,subprocess,sys,time
from pathlib import Path
D=Path('/home/chenyiteng/deployment-20261029/single-extend400-sz1-g67-retention')
N=D.with_name('single-extend400-fqlfix-sz1-g67');R=N/'source'
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,same,exact_signal
assert os.getuid()==1003 and read(N/'readiness.json')['tests_passed']
old=read(D/'identity.json');previous=read(N/'formal-plan.json')['predecessor_identity']
assert same(old) and same(previous) and not (N/'identity.json').exists()
assert read(D/'training-queue-status.json')['phase']=='PREPARED_WAIT_HANDOFF' and not (D/'handoff-complete.json').exists()
rp=read(S/'plan.json');old45=copy.deepcopy({g:rp['slots'][g] for g in ['4','5']})
assert all(rp['slots'][g]['external_identity']==old for g in ['6','7'])
with (N/'queue.log').open('x') as log:
 child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_distillation/single_formal_queue.py'),str(N)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
deadline=time.monotonic()+10
while not (N/'identity.json').exists() and time.monotonic()<deadline:
 assert child.poll() is None;time.sleep(.1)
me=read(N/'identity.json');assert same(me) and me['pid']==child.pid
save(N/'rlt-plan-before.json',rp)
for g in ['6','7']:rp['slots'][g].update(external_identity=me,external_release=str(N/'released.json'))
assert {g:rp['slots'][g] for g in ['4','5']}==old45
save(S/'plan.json',rp)
save(N/'registered.json',dict(time=time.time(),identity=me,waiting_for=previous,gpus=[6,7],operation='CP-v2 400 replaces CP400; FQL-fix200 appended before RLT'))
assert read(D/'training-queue-status.json')['phase']=='PREPARED_WAIT_HANDOFF' and same(previous)
exact_signal(old,signal.SIGTERM)
deadline=time.monotonic()+10
while same(old) and time.monotonic()<deadline:time.sleep(.1)
assert not same(old) and same(me) and same(previous)
receipt=dict(time=time.time(),owner=me,previous_waiting_owner=old,old_owner_stopped=True,predecessor_training_alive=True,slots45_unchanged=True,rlt_owner_alive=same(read(S/'owner-identity.json')),methods=read(N/'formal-plan.json')['methods'],fql_budget_epochs=200,output=read(N/'prepared.json')['output'])
save(N/'activation.json',receipt)
print(json.dumps(receipt))
