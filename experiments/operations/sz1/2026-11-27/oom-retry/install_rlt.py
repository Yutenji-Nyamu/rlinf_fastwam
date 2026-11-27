from pathlib import Path
import fcntl,os,json,sys,shutil
from omegaconf import OmegaConf
R=Path(os.environ["QUEUE_DEPLOYMENT"])
Q=Path(os.environ["QUEUE_ROOT"])
G=R/'oom_retry_v1'
sys.path.insert(0,str(Q/'ops'))
sys.path.insert(0,str(G))
from common import read,save,same
from patch_formal_owners import update
from rlt_formal_oom_1127 import select_retry
def replace(p,data):
    t=p.with_name('.'+p.name+'.oom-installing')
    with t.open('xb') as f:f.write(data)
    if p.is_file():os.chmod(t,p.stat().st_mode)
    t.replace(p)

def main():
    assert os.getuid()==1003
    queue=read(R/'queue.json')['deployment']
    with (Q/'priority-handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        configs={}; owners={}; rows=[]
        for key,chain in queue.items():
            for folder in map(Path,chain[1:]):
                file=folder/'plan.json'
                p=read(file)
                assert p['gpu']==int(key) and p['method']=='rlt' and not p.get('retry_formal_gpu_oom')
                assert same(read(folder/'identity.json')) and 'WAIT' in read(folder/'status.json')['phase']
                cfg=OmegaConf.load(folder/'config/formal.yaml');ok,cp=select_retry(cfg)
                assert ok, folder
                name='rlt_dojo_owner.py' if p['bench']=='dojo' else 'rt2_owner.py'
                owner=Path(p['training_source'])/'tools/dsrl_rlt_warmstart'/name
                if owner not in owners:
                    src=owner.read_text(encoding='utf-8-sig');patched=update(src,'dojo' if p['bench']=='dojo' else 'rt2')
                    compile(patched,str(owner),'exec')
                    owners[owner]=patched.encode()
                configs[file]=dict(p,retry_formal_gpu_oom=True)
                rows.append({'slot':key,'task':p['task'],'checkpoint_now':cp.name if cp else 'original initialization','pending':True})
        assert len(configs)==7 and len(owners)==3
        helpers={Q/'ops'/k:(G/k).read_bytes() for k in ('rlt_formal_oom_1127.py','oom_classify_1127.py')}
        b=G/'rlt-install-backup';b.mkdir(exist_ok=True);backups={}
        try:
            for i,(p,body) in enumerate([*helpers.items(),*owners.items()]):
                backups[p]=p.read_bytes() if p.is_file() else None
                if backups[p] is not None:
                    orig=b/(str(i)+'-'+p.name);orig.write_bytes(backups[p])
                replace(p,body)
            for i,(p,new) in enumerate(configs.items()):
                backups[p]=p.read_bytes();(b/('job-'+str(i)+'.json')).write_bytes(backups[p]);save(p,new)
                assert read(p)['retry_formal_gpu_oom']
            save(G/'formal-applied.json',dict(formal=rows));print('APPLIED',json.dumps(rows))
        except BaseException:
            for p,body in reversed(list(backups.items())):
                if body is None:p.unlink(missing_ok=True)
                else:replace(p,body)
            raise
if __name__=='__main__':main()

