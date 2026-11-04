"""Single GPU: full-noise Clean -> full-noise + U -> existing RLT.

Only task-private drivers and their accepted namespace cleanup are signalled.
Failed methods leave an explicit receipt and advance, never block the queue.
"""
import os,signal,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,'/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops')
from common import read,save,proc,same,exact_signal,gpus

def main(directory):
    d=Path(directory);p=read(d/'plan.json');me=proc(os.getpid());child=None;active=None
    save(d/'identity.json',me);save(d/'status.json',dict(phase='WAIT_PREPARED_HANDOFF',time=time.time()))
    def stop(*_):raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop)
    result={}
    try:
        if p.get('from_rlt'):
            sys.path.insert(0,str(Path(p['training_source'])/'tools/rlt_disk_fallback'))
            from priority_pair_handoff import handoff
            h=d/'rlt-handoff';h.mkdir()
            handoff(h,[p['gpu']],me,reservations={str(p['gpu']):dict(identity=me,release=str(d/'released.json'))})
            save(d/'registered.json',dict(identity=me,time=time.time()))
        while not (d/'registered.json').exists():time.sleep(.1)
        assert read(d/'registered.json')['identity']==me
        deadline=time.monotonic()+300
        while any(same(w) for w in p['predecessors']) or gpus()[p['gpu']]['processes']:
            assert time.monotonic()<deadline,'Previous task not fully cleaned'
            time.sleep(.25)
        failed=set()
        for item in p['phases']:
            role,phase=item['role'],item['phase']
            if role in failed:continue
            active=Path(item['deployment']);ap=read(active/'plan.json')
            env=dict(os.environ,**read(active/'training-environment.json'))
            env.update(DOJO_GRPO_DEPLOYMENT=str(active),ROBOTWIN_DSRL_OWNER=str(d))
            save(d/'status.json',dict(phase=phase,role=role,time=time.time(),deployment=str(active)))
            command=[ap['python'],'-u','-B',str(Path(ap['training_source'])/'tools/dojo_offpolicy/training_entry.py'),'--config-path',str(active/'config'),'--config-name',phase]
            try:
                with (active/f'{phase}.log').open('x') as log:
                    child=subprocess.Popen(command,cwd=ap['training_source'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    who=proc(child.pid);save(d/'child.json',who)
                    code=child.wait(timeout=item.get('timeout'))
                if code:raise RuntimeError(f'{role}/{phase} exit {code}')
                save(d/f'{role}-{phase}-complete.json',dict(time=time.time()))
            except (Exception,KeyboardInterrupt):
                if child and child.poll() is None:
                    exact_signal(who,signal.SIGTERM);child.wait(timeout=180)
                failed.add(role)
                save(d/f'{role}-failed.json',dict(time=time.time(),phase=phase,error=traceback.format_exc()))
            finally:
                if child is not None and child.poll() is not None:child=None
        result=dict(phase='COMPLETE_RETURN_RLT',failed=sorted(failed),time=time.time())
    except BaseException:
        result=dict(phase='FAILED_RETURN_RLT',error=traceback.format_exc(),time=time.time())
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if child and child.poll() is None:
            exact_signal(who,signal.SIGTERM)
            try:child.wait(timeout=180)
            except subprocess.TimeoutExpired:exact_signal(who,signal.SIGKILL)
        save(d/'status.json',result);save(d/'released.json',result)

if __name__=='__main__':main(sys.argv[1])
