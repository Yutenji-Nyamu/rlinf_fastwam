"""Activate shared-GPU OOM recovery on a live queue, without restarting the training owners."""
import copy,fcntl,os,signal,subprocess as sp,sys,time
from pathlib import Path
V=Path(__file__).resolve().parent
G=V.parent
R=G.parent
Q=Path(os.environ['QUEUE_ROOT'])
F=Path(os.environ['FASTWAM_OWNER_ROOT'])
sys.path[:0]=[str(V),str(Q/'ops')]
from common import read,save,same
from patch_formal_owners import update as rlt_update
from patch_dojo_admission import update as dojo_update

def write_code(path,data):
    compile(data,str(path),'exec')
    tmp=path.with_name('.'+path.name+'.contention.tmp')
    with tmp.open('xb') as st:st.write(data)
    os.chmod(tmp,path.stat().st_mode if path.exists() else 0o644)
    os.replace(tmp,path)

def arm(pair,kind,cards,successors,report):
    old=[read(w/'wait.json') for w in successors]
    d=old[0]['oom_retry_gate']
    assert all(row['oom_retry_gate']==d for row in old)
    oldgate=Path(d['release']).parent
    origin=read(pair/'identity.json')
    gate=V/('pair'+''.join(cards))
    if oldgate==gate:
        assert same(d['identity']) and same(origin)
        return
    assert same(d['identity']) and same(origin)
    def healthy():
        return (same(origin) and not (pair/'released.json').exists()
                and read(oldgate/'status.json')['phase']=='MONITORING_ORIGINAL')
    assert healthy()
    gate.mkdir()
    with (gate/'outer.log').open('x') as log:
        child=sp.Popen([sys.executable,'-uB',str(V/'pair_gate.py'),str(gate),str(pair),kind],
                       stdin=sp.DEVNULL,stdout=log,stderr=log,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
    try:
        for _ in range(120):
            if (gate/'identity.json').is_file():break
            if child.poll() is not None:raise RuntimeError((gate,(gate/'outer.log').read_text()[-500:]))
            time.sleep(.1)
        who=read(gate/'identity.json');assert who['pid']==child.pid and same(who)
        assert healthy() and same(d['identity'])
        os.kill(d['identity']['pid'],signal.SIGSTOP)  # Only the old CPU observer.
        try:
            assert healthy()
            save(gate/'registered.json',{'identity':who})
            for _ in range(120):
                if read(gate/'status.json')['phase']=='MONITORING_ORIGINAL':break
                time.sleep(.1)
            assert read(gate/'status.json')['phase']=='MONITORING_ORIGINAL'
            for w,row in zip(successors,old):
                save(w/'wait.json',dict(row,oom_retry_gate=dict(identity=who,release=str(gate/'released.json'),primary_identity=str(pair/'identity.json'))))
            assert all(read(w/'wait.json')['oom_retry_gate']['identity']==who for w in successors)
        except BaseException:
            for w,row in zip(successors,old):save(w/'wait.json',row)
            raise
        else:
            os.kill(d['identity']['pid'],signal.SIGTERM)
        finally:
            if same(d['identity']):os.kill(d['identity']['pid'],signal.SIGCONT)
        assert same(origin)
        report.append(dict(kind=kind,cards=cards,original_training_owner_live=True))
    except BaseException:
        if child.poll() is None:child.terminate()
        raise

def main():
    assert R.stat().st_uid==os.getuid() and os.getuid()!=0
    report=[]
    with (Q/'priority-handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=read(R/'queue.json')['deployment']; pairs=[]; owners={}; changes={}
        for a,b,kind in [('4','7','fastwam'),('5','6','dojo')]:
            pair=Path(queue[a][0]);assert pair==Path(queue[b][0]) and same(read(pair/'identity.json'))
            pairs.append((pair,kind,[a,b],[Path(queue[g][1]) for g in (a,b)]))
            if kind=='dojo':
                src=Path(read(pair/'plan.json')['training_source'])/'tools/dojo_grpo/pair_owner.py'
                changes[src]=dojo_update(src.read_text()).encode()
        for g,items in queue.items():
            for job in map(Path,items[1:]):
                assert same(read(job/'identity.json')) and 'WAIT' in read(job/'status.json')['phase']
                plan=read(job/'plan.json');assert plan['retry_formal_gpu_oom'] and plan['gpu']==int(g)
                bench=plan['bench'];name='rlt_dojo_owner.py' if bench=='dojo' else 'rt2_owner.py'
                owners[Path(plan['training_source'])/'tools/dsrl_rlt_warmstart'/name]=bench
        assert len(owners)==3 and sum(len(x)-1 for x in queue.values())==7
        for n in ('oom_classify_1127.py','oom_contention.py','rlt_formal_oom_1127.py'):
            changes[Q/'ops'/n]=(V/n).read_bytes()
        for p,bench in owners.items():changes[p]=rlt_update(p.read_text(encoding='utf-8-sig'),bench).encode()
        for n in ('fastwam_pair.py','queue_admission.py'):
            p=F/n;assert p.read_bytes() in ((G/n).read_bytes(),(V/n).read_bytes());changes[p]=(V/n).read_bytes()
        for path,data in changes.items():compile(data,str(path),'exec')
        backup=V/'before';backup.mkdir(exist_ok=True)
        installed=[]
        try:
            for idx,(path,data) in enumerate(changes.items()):
                before=path.read_bytes() if path.exists() else None
                if before==data:continue
                if before is not None:
                    dest=backup/(str(idx)+'-'+path.name)
                    if not dest.exists():dest.write_bytes(before)
                write_code(path,data);installed.append((path,before))
        except BaseException:
            for path,before in reversed(installed):
                if before is None:path.unlink()
                else:write_code(path,before)
            raise
        for entry in pairs:arm(*entry,report)
        assert all(p.read_bytes()==data for p,data in changes.items())
        print('VERIFIED',report,'RLT plans=7, RLT launchers=3; both original training owners are alive',flush=True)
if __name__=='__main__':main()

