import copy,json,time,os,fcntl,subprocess as sp,sys,shutil
from pathlib import Path
from tempfile import TemporaryDirectory
R=Path(os.environ["QUEUE_DEPLOYMENT"])
Q=Path(os.environ["QUEUE_ROOT"])
sys.path.insert(0,str(Q/'ops'))
from common import read,save,same,proc
G=R/'oom_retry_v1'
F=Path(os.environ["FASTWAM_OWNER_ROOT"])
def atomic_code(dst,src):
    old=dst.read_bytes();new=src.read_bytes()
    compile(new,str(dst),'exec')
    tmp=dst.with_name('.'+dst.name+'.oom-retry.tmp')
    with tmp.open('xb') as h:h.write(new)
    os.chmod(tmp,dst.stat().st_mode);tmp.replace(dst)
def arm():
    assert os.getuid()==1003
    with (Q/'priority-handoff.lock').open('a') as handoff:
        fcntl.flock(handoff,fcntl.LOCK_EX)
        c=read(R/'queue.json')['deployment']
        queue=read(Q/'plan.json')
        assert str(queue['boot_id'])==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        pairs=[('47','fastwam','4','7'),('56','dojo','5','6')]
        original=F/'queue_admission.py'
        assert original.read_text(encoding='utf-8-sig')==(G/'queue_admission.original.py').read_text(encoding='utf-8-sig')
        state=read(Q/'status.json')['slots'];adoptions=[];backups={};guards=[]
        for name,kind,a,b in pairs:
            p=Path(c[a][0]);gate=G/('pair'+name)
            assert p==Path(c[b][0]) and not gate.exists()
            assert same(read(p/'identity.json')) and not (p/'released.json').exists()
            assert all(state[g]['phase']=='WAIT_EXTERNAL' for g in (a,b))
            assert all(queue['slots'][g]['reserved_sequence']==c[g] for g in (a,b))
            firsts=[Path(c[g][1]) for g in (a,b)]
            for w in firsts:
                row=read(w/'wait.json')
                assert (str(p/'released.json') in [v['release'] for v in row['blockers']])
                assert 'oom_retry_gate' not in row
                who=read(w/'identity.json')
                assert same(who) and 'WAIT' in read(w/'status.json')['phase']
                cmd=(Path('/proc')/str(who['pid'])/'cmdline').read_bytes().split(b'\0')
                assert os.fsencode(F/'wait_owner.py') in cmd and os.fsencode(w) in cmd
                backups[w]=row
            adoptions.append((gate,p,kind,firsts))
        fast_launcher=F/'fastwam_pair.py'
        assert fast_launcher.read_bytes()==(G/'fastwam_pair.before.py').read_bytes()
        atomic_code(original,G/'queue_admission.py')
        try:
            atomic_code(fast_launcher,G/'fastwam_pair.py')
            for gate,p,kind,ws in adoptions:
                gate.mkdir()
                with (gate/'outer.log').open('x') as f:
                    cp=sp.Popen([sys.executable,'-uB',str(G/'pair_gate.py'),str(gate),str(p),kind],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=f,stdin=sp.DEVNULL,start_new_session=True)
                guards.append(cp)
                for _ in range(120):
                    if (gate/'identity.json').is_file():break
                    if cp.poll() is not None:raise RuntimeError((gate,(gate/'outer.log').read_text()[-600:]))
                    time.sleep(.1)
                who=read(gate/'identity.json')
                assert who['pid']==cp.pid and same(who)
                desc={'identity':who,'release':str(gate/'released.json'),'primary_identity':str(p/'identity.json')}
                for w in ws:
                    data=copy.deepcopy(backups[w]);data['oom_retry_gate']=desc
                    save(w/'wait.json',data);assert read(w/'wait.json')['oom_retry_gate']==desc
            for gate,p,kind,ws in adoptions:
                assert same(read(p/'identity.json')) and not (p/'released.json').exists()
                save(gate/'registered.json',{'identity':read(gate/'identity.json')})
            for gate,p,kind,ws in adoptions:
                assert same(read(gate/'identity.json'))
                print('ARMED',kind,read(p/'identity.json')['pid'],read(gate/'identity.json')['pid'],[w.name for w in ws])
        except BaseException:
            for w,row in backups.items():save(w/'wait.json',row)
            for g in guards:
                if g.poll() is None:g.terminate()
            atomic_code(original,G/'queue_admission.original.py')
            atomic_code(fast_launcher,G/'fastwam_pair.before.py')
            raise
if __name__=='__main__':arm()

