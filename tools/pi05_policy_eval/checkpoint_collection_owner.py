"""Collection-only successor to an intentionally stopped GRPO run.

Reuse the accepted fixed-pool collector and precise token cleanup. The FQL
successor waits on this owner, including failure, before releasing to RLT.
"""
import json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, '/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops')
from common import read, save, proc, same, exact_signal, gpus, admission_card

def main(directory):
    d=Path(directory);p=read(d/'plan.json');me=proc(os.getpid());child=None
    save(d/'identity.json',me)
    save(d/'status.json',dict(phase='WAIT_PREPARED_HANDOFF',time=time.time()))
    def stop(*_): raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop)
    try:
        while not (d/'registered.json').exists(): time.sleep(.1)
        assert read(d/'registered.json')['identity']==me
        deadline=time.monotonic()+600
        while any(same(w) for w in p['predecessors']) or any(admission_card(gpus()[g],p.get('gpu_admission',{}).get(str(g),{}))['processes'] for g in p['gpus']):
            assert time.monotonic()<deadline,'Predecessor cleanup incomplete'
            time.sleep(.25)
        save(d/'status.json',dict(phase='FIXED_POOL_AND_REPLAY',time=time.time(),milestones=p['milestones']))
        with (d/'collection.log').open('x') as log:
            child=subprocess.Popen([p['python'],'-u','-B',str(d/'tools/six_point_after.py'),str(d)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            who=proc(child.pid);save(d/'child.json',who)
            rc=child.wait(timeout=14400)
        assert rc==0, f'Collection exited {rc}'
        result=dict(phase='COMPLETE',time=time.time(),output=p['output'])
    except BaseException:
        result=dict(phase='FAILED_ADVANCE_FQL',time=time.time(),error=traceback.format_exc())
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if child and child.poll() is None:
            exact_signal(who,signal.SIGTERM)
            try: child.wait(timeout=60)
            except subprocess.TimeoutExpired: exact_signal(who,signal.SIGKILL)
        save(d/'status.json',result);save(d/'released.json',result)

if __name__=='__main__': main(sys.argv[1])
