"""One-shot GPU4 return repair: shorter output path, same frozen CP/config.

Preserves the original failed return receipts. Never calls stop or changes GPU567.
"""
import copy,fcntl,hashlib,importlib.util,json,os,socket,sys
from pathlib import Path
from omegaconf import OmegaConf
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D=S/'rynn-numeric-v1';OLD=D/'prepared/rynn-numeric-v1-gpu4'
ROOT=D/'rlt-return-repair-v1';NEW=ROOT/OLD.name
RUN=Path('/data/chenyiteng/results/rlinf-rlt/rlt-sz3-g4-cp225-1005-num-return-v1')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    p.chmod(0o600)

def main():
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    final=read(D/'run/final.json')
    assert final['terminal_status']=='completed' and final['child_exit_code']==0
    assert final['rlt_borrowed'] and not final['rlt_return_dispatched']
    assert final['recovery_error']['type']=='OSError' and '[Errno 36]' in final['recovery_error']['error']
    assert read(D/'run/cleanup.json')['all_stopped']
    R=load('numeric_return_repair_original',OLD/'rlt_returned_cycle.py');R.install_helper(OLD)
    H=R.H;p=R.load_plan(OLD);stopped=read(OLD/'rlt-stopped.json')
    assert set(p['runs'])=={'gpu4'} and p['runs']['gpu4']['gpus']==[4]
    assert not H.same(read(D/'run/owner-identity.json'))
    untouched=read(D/'run/owner-plan.json')['untouched_drivers']
    assert all(H.same(v['identity']) for v in untouched.values())
    assert not ROOT.exists() and not RUN.exists() and len(RUN.name.encode())<100
    with (OLD/'operation.lock').open('a') as oldlock:
        fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any((OLD/n).exists() for n in ('gpu4-launch-attempt.json','gpu4-launched.json','resumed-dispatched.json'))
        live=H.actors(p);row=p['runs']['gpu4']
        assert not H.same(row['original_identity']) and not H.active(live,row['original_namespace'])
        assert not H.active(live,row['namespace']) and not H.gpu_processes([4])
        release=D/'run/diagnostic-release.json';verified=H.validate_release(OLD,release)
        assert verified==read(OLD/'dojo-release-verified.json')
        ROOT.mkdir(mode=0o700);NEW.mkdir(mode=0o700);pre=NEW/'prepared/gpu4';pre.mkdir(parents=True)
        for n in ('rlt_returned_cycle.py','base_rlt_gpu567_cycle.py','rlt_checkpoint_lifecycle.py','next_six_ops.py'):
            (NEW/n).write_bytes((OLD/n).read_bytes())
        oldpre=OLD/'prepared/gpu4'
        assert sha(oldpre/'original.yaml')==row['prepared_sha256']['original.yaml']
        assert sha(oldpre/'environment.json')==row['prepared_sha256']['environment.json']
        assert sha(oldpre/'resolved.yaml')==stopped['runs']['gpu4']['resolved_sha256']
        original=H.config(oldpre/'original.yaml')
        recovery=stopped['runs']['gpu4']['recovery'];assert recovery['checkpoint']['step']==225
        cfg,changes=H.resumed_config(original,row['original_run'],RUN,recovery['checkpoint']['path'])
        (pre/'original.yaml').write_bytes((oldpre/'original.yaml').read_bytes())
        OmegaConf.save(OmegaConf.create(cfg),str(pre/'resolved.yaml'))
        env=H.read(oldpre/'environment.json')
        env={k:v.replace(row['new_run'],str(RUN)) if isinstance(v,str) else v for k,v in env.items()}
        assert not any(k in env for k in H.MASKS)
        save(pre/'environment.json',env)
        q=copy.deepcopy(p);q['time']=H.now();q['runs']['gpu4'].update(new_run=str(RUN),config_changes=changes,recovery=recovery)
        q['runs']['gpu4']['prepared_sha256']={n:sha(pre/n) for n in ('original.yaml','resolved.yaml','environment.json')}
        evidence={str(f):sha(f) for f in (OLD/'plan.json',OLD/'rlt-stopped.json',D/'run/final.json',D/'run/cleanup.json',release,Path(__file__))}
        q['frozen_files'].update(evidence)
        q['repair']={'reason':'ENAMETOOLONG before any driver dispatch','original_stage':str(OLD),'same_cycle_id_different_path':True,'no_new_stop':True,'evidence':evidence}
        save(NEW/'plan.json',q)
        newstopped=copy.deepcopy(stopped)
        newstopped['runs']['gpu4']['resolved_sha256']=sha(pre/'resolved.yaml')
        newstopped['runs']['gpu4']['config_changes']=changes
        newstopped['repair_no_new_stop']=True;newstopped['derived_from']={'path':str(OLD/'rlt-stopped.json'),'sha256':sha(OLD/'rlt-stopped.json')}
        save(NEW/'rlt-stopped.json',newstopped)
        save(ROOT/'prepared.json',dict(time=H.now(),old_stage=str(OLD),new_stage=str(NEW),short_run=str(RUN),
            same_checkpoint=recovery['checkpoint']['path'],plan_sha256=sha(NEW/'plan.json'),config_changes=changes,evidence=evidence))
        NR=load('numeric_return_repair_derived',NEW/'rlt_returned_cycle.py');NR.install_helper(NEW)
        NR.load_plan(NEW)
        with (NEW/'operation.lock').open('a') as newlock:
            fcntl.flock(newlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            answer=NR.H.resume(NEW,release)
        assert answer['resumed_dispatched']
        assert all(NR.H.same(v['identity']) for v in untouched.values())
        receipt=dict(time=NR.H.now(),result=answer,lifecycle_path=str(NEW),new_run=str(RUN),
            original_final_preserved_sha256=sha(D/'run/final.json'),plan_sha256=sha(NEW/'plan.json'),
            return_sha256=sha(NEW/'resumed-dispatched.json'),gpu4_launch=read(NEW/'gpu4-launched.json'),
            untouched_drivers_live={k:NR.H.same(v['identity']) for k,v in untouched.items()})
        save(ROOT/'repaired.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
