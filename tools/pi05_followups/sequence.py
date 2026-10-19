"""Chain accepted finite lanes before the unchanged per-card RLT owner."""
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same,exact_signal,gpus

D=Path(sys.argv[1]);p=read(D/'prepared.json');R=Path(p['source'])
ROOTS={k:Path(v)for k,v in p['roots'].items()}
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
assert os.getuid()==p['uid']==1003
me=proc(os.getpid());children=[]
token=D.name
save(D/'sequence-identity.json',me)


def stop(*_):
    for child in children:
        if child.poll()is None:exact_signal(proc(child.pid),signal.SIGTERM)
    raise SystemExit(143)


signal.signal(signal.SIGTERM,stop)


def clearance():
    owned=[]
    for card in gpus().values():
        for row in card['processes']:
            try:
                environ=Path(f"/proc/{row['pid']}/environ").read_text().split('\0')
            except OSError:continue
            if f'FOLLOWUP_TOKEN={token}'in environ:
                identity=proc(row['pid'])
                assert identity['uid']==os.getuid()
                if identity not in owned:owned.append(identity)
    for identity in owned:exact_signal(identity,signal.SIGTERM)
    if owned:time.sleep(3)
    snapshot=gpus()
    return {'time':time.time(),'cards_clear':{str(g):not snapshot[g]['processes']for g in (4,5)},'signalled':owned}


def phase(name,kind,root):
    global children
    transaction=D/name;children=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',FOLLOWUP_TOKEN=token)
    if kind=='noise':
        env.update(U_VIDEO_ROOT=str(root),U_VIDEO_TRANSACTION=str(transaction))
        if name=='noise-full':env['U_VIDEO_ACCEPTED_TRANSACTION']=str(D/'noise-smoke')
        script=R/'tools/pi05_u_video/lane.py'
    else:
        env.update(POLICY_EVAL_ROOT=str(root),POLICY_EVAL_TRANSACTION=str(transaction),
                   POLICY_EVAL_UID='1003',POLICY_EVAL_GPUS='4,5')
        script=R/'tools/pi05_policy_eval/lane.py'
    save(D/'status.json',{'time':time.time(),'phase':name,'root':str(root)})
    for g in (4,5):
        with(transaction/f'lane-g{g}.log').open('x')as log:
            child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(script),str(g)],
                    cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
    while any(child.poll()is None for child in children):
        if any(child.poll() not in (None,0)for child in children):
            for child in children:
                if child.poll()is None:exact_signal(proc(child.pid),signal.SIGTERM)
            raise RuntimeError(f'{name} lane failed; no dependent phase started')
        time.sleep(10)
    assert all(child.returncode==0 for child in children),name
    receipts=[read(transaction/f'lane-g{g}-released.json')for g in (4,5)]
    assert all(row['completed']for row in receipts),name
    assert all(not gpus()[g]['processes']for g in (4,5)),name
    save(D/f'{name}-complete.json',{'time':time.time(),'receipts':receipts})


try:
    old=p['plan_before']['slots']['4']
    save(D/'status.json',{'time':time.time(),'phase':'WAIT_PREDECESSOR'})
    while same(old['external_identity']) or not Path(old['external_release']).exists():time.sleep(10)
    release=read(old['external_release']);assert release['owned_contexts_clear'] and release['cleanup']['ok']
    assert all(not gpus()[g]['processes']for g in (4,5))
    phase('noise-smoke','noise',ROOTS['noise'])
    for g in (4,5):assert read(D/'noise-smoke'/f'acceptance-g{g}.json')['validation']['passed']
    phase('switch-smoke','policy',ROOTS['smoke'])
    save(D/'smoke-complete.json',{'time':time.time(),'noise':True,'early2':True,'c20':True})
    save(D/'status.json',{'time':time.time(),'phase':'WAIT_PUBLICATION_AND_ACCEPTANCE'})
    # One local acceptance gate after the real smoke. No GPU context is held.
    while not(D/'continue.json').exists():time.sleep(10)
    phase('noise-full','noise',ROOTS['noise'])
    phase('early','policy',ROOTS['early'])
    phase('c20-rank','policy',ROOTS['rank'])
    with(D/'c20-prepare.log').open('x')as log:
        child=subprocess.Popen([PY,'-u','-B',str(R/'tools/pi05_followups/prepare.py'),'c20'],
              cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PI05_FOLLOWUP_VERSION=p.get('version','v1')),stdout=log,stderr=subprocess.STDOUT)
        children=[child];assert child.wait(timeout=300)==0,'C20 calibration/selection preparation failed'
    if not(D/'c20-not-stronger.json').exists():phase('c20-sweep','policy',ROOTS['c20'])
    clear=clearance();assert all(clear['cards_clear'].values())
    save(D/'released.json',{'time':time.time(),'identity':me,'completed':True,'clearance':clear})
    save(D/'status.json',{'time':time.time(),'phase':'COMPLETE_RLT_HANDOFF'})
except BaseException:
    save(D/'error.json',{'time':time.time(),'error':traceback.format_exc()})
    for child in children:
        if child.poll()is None:
            exact_signal(proc(child.pid),signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:pass
    clear=clearance()
    # The existing owner additionally checks process identity and all GPU C/G
    # contexts before returning each card; a failure cannot bypass that check.
    save(D/'released.json',{'time':time.time(),'identity':me,'completed':False,'clearance':clear})
    raise
