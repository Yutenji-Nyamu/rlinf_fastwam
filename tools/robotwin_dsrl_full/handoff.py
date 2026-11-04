"""Prepared GPU6 repeat-Dojo -> RoboTwin queue; keep unrelated work intact."""
import fcntl,json,os,signal,sys,time
from pathlib import Path
B=Path('/data/chenyiteng/deployment-20261104')
R=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rt2-pi05-dsrl-full-20261104')
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
sys.path.insert(0,str(S/'ops'));from common import read,save,proc,same,exact_signal
sys.path.insert(0,str(R));from tools.dojo_offpolicy.swap_4567 import launch

def main():
    assert os.getuid()==1003
    d=B/'rt2-dsrl-full-g6';old=B/'dojo-dsrl-repeat-g6-formal'
    if (old/'recovery-current.json').exists():old=Path(read(old/'recovery-current.json')['deployment'])
    oldwho=read(old/'identity.json');assert same(oldwho)
    p=read(d/'plan.json');p['predecessors']=[oldwho];save(d/'plan.json',p)
    assert '3 passed' in (d/'cpu-tests.txt').read_text()
    new=launch(d,[PY,'-u','-B',str(R/'tools/robotwin_dsrl_full/queue.py'),str(d)],cwd=R)
    serial=(S/'priority-handoff.lock').open('a');fcntl.flock(serial,fcntl.LOCK_EX)
    owner=read(S/'owner-identity.json');exact_signal(owner,signal.SIGTERM)
    deadline=time.monotonic()+30
    while same(owner):
        assert time.monotonic()<deadline;time.sleep(.1)
    lock=(S/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan,state=read(S/'plan.json'),read(S/'status.json');save(d/'previous-rlt-plan.json',plan)
    plan['slots']['6'].update(external_identity=new,external_release=str(d/'released.json'),external_blockers=[])
    state['slots']['6']={'phase':'WAIT_EXTERNAL'}
    save(S/'plan.json',plan);save(S/'status.json',state)
    # Prevent the old Dojo watchdog from recovering the intentionally ended run.
    watcher=read(B/'dojo-dsrl-repeat-g6-formal/recovery-identity.json')
    if same(watcher):exact_signal(watcher,signal.SIGTERM)
    save(d/'registered.json',dict(identity=new,time=time.time()))
    lock.close();launch(d,['/usr/bin/python3','-u','-B',str(S/'ops/owner.py'),'--plan',str(S/'plan.json')],name='rlt-owner.log',identity=None,cwd=S);serial.close()
    exact_signal(oldwho,signal.SIGTERM)
    receipt=dict(time=time.time(),old_deployment=str(old),old_owner=oldwho,old_watch=watcher,new_owner=new,gpu=6,order=['clean smoke/resume','u smoke/resume','Clean200','U200','RLT'],reason='user ends repeated-noise Dojo; turn_switch pi05 full-noise DSRL')
    save(d/'handoff.json',receipt);save(old/'intentional-stop.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':main()
