import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
D=Path('/home/chenyiteng/deployment-20261101/gad-k4-v1')
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal
from priority_pair_handoff import handoff
R=D/'source';children={}
for pair in ['45','67']:
    pd=D/f'pair{pair}'
    assert not (pd/'identity.json').exists(),'No duplicate queue owners'
    with (pd/'queue.log').open('x') as log:
        child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(R/'tools/pi05_distillation/gad_queue.py'),str(pd)],
           cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R)),
           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.time()+30
    while not (pd/'identity.json').exists():
        assert child.poll() is None and time.time()<deadline,'Queue startup failed before GPU admission'
        time.sleep(.1)
    children[pair]=read(pd/'identity.json')
receipt=handoff(D/'pair67',[6,7],children['67'])
# 4/5 are behind a healthy existing FQL owner. Only replace their future tail;
# keep that predecessor identity as a blocker, never signal its process.
pd=D/'pair45'
with (S/'priority-handoff.lock').open('a') as serial:
    fcntl.flock(serial,fcntl.LOCK_EX)
    owner=read(S/'owner-identity.json');exact_signal(owner,signal.SIGTERM)
    deadline=time.time()+30
    while same(owner):
        assert time.time()<deadline;time.sleep(.1)
    with (S/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(S/'plan.json');state=read(S/'status.json')
        save(pd/'rlt-plan-before.json',p)
        for g in ['4','5']:
            slot=p['slots'][g];assert state['slots'][g]['phase']=='WAIT_EXTERNAL'
            predecessor=slot['external_identity']
            assert predecessor==read(pd/'formal-plan.json')['predecessor_identities'][0]
            slot.update(external_identity=children['45'],external_release=str(pd/'released.json'),
                        external_blockers=[predecessor])
        save(S/'plan.json',p);save(S/'status.json',state)
    with (pd/'owner.log').open('x') as log:
        ownerchild=subprocess.Popen(['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],
            cwd=S,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(pd/'registered.json',dict(time=time.time(),identity=children['45'],predecessors_retained=True))
save(D/'activation.json',dict(time=time.time(),queues=children,rlt67=receipt,owner=proc(ownerchild.pid)))
print(json.dumps(read(D/'activation.json')))
