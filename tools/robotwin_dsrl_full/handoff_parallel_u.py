"""Prepared FQL -> independent U jobs; single-card RLT until both finish."""
import fcntl,os,signal,sys,time,shutil
from pathlib import Path
B=Path('/data/chenyiteng/deployment-20261104')
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/parallel-dsrl-u-20261104')
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
sys.path.insert(0,str(S/'ops'));from common import read,save,same,exact_signal
sys.path.insert(0,str(R));from tools.dojo_offpolicy import swap_4567 as swap
swap.R=R;launch=swap.launch

def main():
    assert os.getuid()==1003
    pair=B/'parallel-u47';old=B/'fql-v2-after-fourpoint-g47';f=B/'fql-after-parallel-u47'
    info=read(pair/'prepared.json');jobs=list(map(Path,info['jobs']))
    # Use the same full checkpoint and completed evaluation; no model rewind
    # across deleted rolling points. Preparation can run while FQL remains busy.
    output=Path(read(old/'prepared.json')['output'])
    deadline=time.monotonic()+900
    while True:
        cp=read(output/'fql_fixed_teacher/checkpoint.json')
        summary_path=output/f"summary-fql_fixed_teacher-s{cp.get('step',0):06d}.json"
        summary=read(summary_path) if summary_path.exists() else {}
        if summary.get('complete') and summary.get('checkpoint',{}).get('path')==cp.get('path') and Path(cp['path']).is_file():break
        assert time.monotonic()<deadline,'Current FQL evaluation has not completed; FQL left running'
        time.sleep(2)
    predecessor=read(old/'identity.json');assert same(predecessor)
    f.mkdir(exist_ok=True);assert not (f/'identity.json').exists();shutil.copytree(old/'seeds',f/'seeds',dirs_exist_ok=True)
    for n in ['formal-plan.json','execution-source.json','fql_fixed_teacher.json','eval-template.json','prepared.json','fql-data-receipt.json']:save(f/n,read(old/n))
    fp=read(f/'formal-plan.json');fp.update(initial_receipts={'fql_fixed_teacher':cp},resume_in_progress=['fql_fixed_teacher'])
    save(f/'formal-plan.json',fp)
    owners={}
    for d,g,tool in [(jobs[0],4,'tools/dojo_offpolicy/owner.py'),(jobs[1],7,'tools/robotwin_dsrl_full/queue.py')]:
        p=read(d/'plan.json');p['predecessors']=[predecessor];save(d/'plan.json',p)
        owners[g]=launch(d,[PY,'-u','-B',str(R/tool),str(d)],cwd=R)
    save(pair/'join-plan.json',dict(jobs=list(map(str,jobs)),gpus=[4,7],fql=str(f),source=str(R),python=PY))
    join=launch(pair,[PY,'-u','-B',str(R/'tools/robotwin_dsrl_full/join_fql.py'),str(pair)],cwd=R,name='join.log',identity='join-identity.json')
    fp.update(predecessor_identity=join,predecessor_deployment=str(pair));save(f/'formal-plan.json',fp)
    serial=(S/'priority-handoff.lock').open('a');fcntl.flock(serial,fcntl.LOCK_EX)
    oldowner=read(S/'owner-identity.json');exact_signal(oldowner,signal.SIGTERM)
    until=time.monotonic()+30
    while same(oldowner):assert time.monotonic()<until;time.sleep(.1)
    lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p,s=read(S/'plan.json'),read(S/'status.json');save(pair/'previous-rlt-plan.json',p)
    for d,g in zip(jobs,[4,7]):
        p['slots'][str(g)].update(external_identity=owners[g],external_release=str(d/'released.json'),external_blockers=[])
        s['slots'][str(g)]={'phase':'WAIT_EXTERNAL'}
    save(S/'plan.json',p);save(S/'status.json',s)
    # Future GPU6 U now delegates to GPU7; the already running Clean is untouched.
    moved=B/'rt2-dsrl-full-g6-r3/u';mp=read(moved/'plan.json');mp['defer_formal_to']=str(B/'rt2-dsrl-u-g7');save(moved/'plan.json',mp)
    save(moved/'moved-to-g7.json',dict(time=time.time(),destination=str(B/'rt2-dsrl-u-g7'),reason='user parallel U scheduling'))
    for d,g in zip(jobs,[4,7]):save(d/'registered.json',dict(identity=owners[g],time=time.time()))
    lock.close()
    launch(pair,['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],cwd=S,name='rlt-owner.log',identity=None)
    serial.close()
    exact_signal(predecessor,signal.SIGTERM)
    receipt=dict(time=time.time(),fql_resume=cp,stopped=predecessor,owners=owners,join=join,order='Dojo U / RT2 U independently; early release -> RLT; both released -> FQL -> RLT')
    save(pair/'handoff.json',receipt);save(old/'intentional-pause.json',receipt)
    # Reuse the accepted abrupt-owner cleanup; startup already has its own bound.
    launch(pair,[PY,'-u','-B','-c',f'from tools.dojo_offpolicy.recover_queue import watch_owner; watch_owner({str(jobs[0])!r})'],cwd=R,name='dojo-watch.log',identity=None)
    print(receipt)

if __name__=='__main__':main()

