"""Both independent U jobs finish -> reclaim their RLT slots -> resume FQL.

While only one is done the unchanged RLT owner fills that card. This CPU-only
join does not reserve either GPU before both terminal releases are present.
"""
import os,sys,time,subprocess,traceback
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'))
from common import read,save,proc,same

def both_released(jobs):
    return all((Path(d)/'released.json').exists() and not same(read(Path(d)/'identity.json')) for d in jobs)

def main(directory):
    d=Path(directory);p=read(d/'join-plan.json');me=proc(os.getpid())
    save(d/'join-identity.json',me)
    try:
        save(d/'join-status.json',dict(phase='WAIT_BOTH_U',time=time.time()))
        while not both_released(p['jobs']):time.sleep(2)
        # RLT may be starting on the just released card. Its standard handoff
        # handles running or queued candidates and preserves the full checkpoint.
        while any(read(S/'status.json')['slots'][str(g)]['phase'] not in ('RLT_RUNNING','RLT_EXITED','WAITING_RLT') for g in p['gpus']):time.sleep(1)
        f=Path(p['fql']);root=Path(p['source']);py=p['python']
        with (f/'queue.log').open('x') as log:
            child=subprocess.Popen([py,'-u','-B',str(root/'tools/dojo_offpolicy/fql_queue_admission.py'),str(f),read(f/'execution-source.json')['source']],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root)),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        deadline=time.monotonic()+60
        while not (f/'identity.json').exists():
            assert child.poll() is None and time.monotonic()<deadline
            time.sleep(.1)
        who=read(f/'identity.json')
        sys.path.insert(0,str(root/'tools/rlt_disk_fallback'))
        from priority_pair_handoff import handoff
        h=d/'fql-handoff';h.mkdir()
        handoff(h,p['gpus'],who,reservations={str(g):dict(identity=who,release=str(f/'released.json')) for g in p['gpus']})
        save(f/'registered.json',dict(identity=who,time=time.time()))
        save(d/'join-status.json',dict(phase='FQL_REGISTERED',identity=who,time=time.time()))
    except BaseException:
        save(d/'join-status.json',dict(phase='FAILED_RLT_REMAINS',error=traceback.format_exc(),time=time.time()))
        raise

if __name__=='__main__':main(sys.argv[1])
