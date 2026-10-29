import json,os,subprocess,sys,time
from pathlib import Path
D=Path('/home/chenyiteng/deployment-20261029/single-extend400-sz1-g67');R=D/'source'
B=Path('/home/chenyiteng/deployment-20261028/physical-v2-formal-sz1-g67')
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,same
assert os.getuid()==1003 and all(read(D/'readiness.json')[k] for k in ['tests_passed','entry_help_passed','queue_compiled'])
assert not (D/'identity.json').exists()
previous=read(B/'identity.json');assert same(previous)
rp=read(S/'plan.json');assert all(rp['slots'][g]['external_identity']==previous for g in ['6','7'])
old45={g:dict(rp['slots'][g]) for g in ['4','5']}
assert read(D/'formal-plan.json')['predecessor_identity']==previous
with (D/'queue.log').open('x') as log:
 child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_distillation/single_formal_queue.py'),str(D)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
deadline=time.monotonic()+10
while not (D/'identity.json').exists() and time.monotonic()<deadline:
 assert child.poll() is None;time.sleep(.1)
me=read(D/'identity.json');assert same(me) and me['pid']==child.pid
save(D/'rlt-plan-before.json',rp)
for g in ['6','7']:rp['slots'][g].update(external_identity=me,external_release=str(D/'released.json'))
assert all(rp['slots'][g]==old45[g] for g in ['4','5'])
save(S/'plan.json',rp)
save(D/'registered.json',dict(time=time.time(),identity=me,gpus=[6,7],waiting_for=previous,operation='append four completed old methods from 200 to 400 budget epochs, after CP/DMD/MFD v2 and before existing RLT; no stop signal'))
assert same(previous) and same(me)
print(json.dumps({'queued':True,'owner':me,'predecessor_alive':True,'current':read(B/'training-queue-status.json'),'new_state':read(D/'training-queue-status.json'),'rlt_slots':{g:rp['slots'][g] for g in ['6','7']},'rlt_owner_alive':same(read(S/'owner-identity.json')),'slots45_unchanged':True}))
