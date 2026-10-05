"""One-shot GPU4 return repair: shorter output path, same frozen CP/config.

Preserves the original failed return receipts. Never calls stop or changes GPU567.
"""
import copy,fcntl,hashlib,importlib.util,json,os,socket,sys
from pathlib import Path
from omegaconf import OmegaConf
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D=S/'rynn-numeric-v1';OLD=D/'prepared/rynn-numeric-v1-gpu4'
ROOT=D/'rlt-return-repair-v2';NEW=ROOT/OLD.name
RUN=Path('/data/chenyiteng/results/rlinf-rlt/rlt-sz3-g4-cp225-1005-num-return-v1')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    p.chmod(0o600)

def refreshed_scope(plan):
    old_manifest=Path(plan['scope_manifest']);old_activation=Path(plan['scope_activation'])
    m=read(old_manifest);a=read(old_activation)
    assert sha(old_manifest)==a['manifest_sha256'] and a['status']=='active'
    assert m['physical_gpus']==[4,5,6,7] and m['cpu_full_mask_target']==4
    backup=Path('/data/chenyiteng/deployment-20261005/management-tidy-v1')
    changes={}
    def renamed_only(path,old_hash):
        current=Path(path);prior=backup/(current.name+'.before')
        assert current.stat().st_uid==20001 and not current.is_symlink()
        assert sha(prior)==old_hash
        value=read(current);normalized=copy.deepcopy(value);names=[]
        for rule in normalized['rules']:
            profile=rule['profile'];assert set(profile)=={'name','settings'}
            names.append(profile['name']);rule['profile']=profile['settings']
        assert len(names)==len(set(names)) and normalized==read(prior)
        changes[str(current)]={'prior_sha256':old_hash,'sha256':sha(current),'rules_unchanged':True,'backup':str(prior)}
        return sha(current)
    m['profile_sha256']=renamed_only(m['profile_path'],m['profile_sha256'])
    for path,item in m['existing_profiles'].items():
        if sha(path)!=item['sha256']:
            assert Path(path).name=='00-opendw-g4-opendw20261003v4.json'
            item['sha256']=renamed_only(path,item['sha256']);item['size']=Path(path).stat().st_size
    scope=ROOT/'scope';scope.mkdir();mp=scope/'scope.json';ap=scope/'activation.json';ep=scope/'environment-fragment.json'
    bootstrap=scope/'bootstrap';bootstrap.mkdir();(scope/'receipts').mkdir()
    assert sha(m['runtime_path'])==m['runtime_sha256'] and sha(m['bootstrap_path'])==m['bootstrap_sha256']
    runtime=bootstrap/'graphics_scope_runtime.py';runtime.write_bytes(Path(__file__).with_name('graphics_scope_runtime.py').read_bytes());runtime.chmod(0o600)
    site=bootstrap/'sitecustomize.py';site.write_bytes(Path(m['bootstrap_path']).read_bytes());site.chmod(0o600)
    m['runtime_path']=str(runtime);m['runtime_sha256']=sha(runtime)
    m['bootstrap_path']=str(site);m['bootstrap_sha256']=sha(site);m['receipts_dir']=str(scope/'receipts')
    save(mp,m)
    a['derived_from']={'path':str(old_activation),'sha256':sha(old_activation),'reason':'Only profile names changed; matching rules and GPU masks verified identical.'}
    a['manifest']=str(mp);a['manifest_sha256']=sha(mp);a['profile_sha256']=m['profile_sha256']
    a['runtime_path']=str(runtime);a['runtime_sha256']=sha(runtime)
    a['activation_receipt']=str(ap)
    a['environment_fragment']['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST']=str(mp)
    oldbootstrap=str(Path(read(old_manifest)['bootstrap_path']).parent)
    tail=[x for x in a['environment_fragment']['PYTHONPATH'].split(':') if x and x!=oldbootstrap]
    a['environment_fragment']['PYTHONPATH']=':'.join([str(bootstrap),*tail])
    save(ep,a['environment_fragment']);a['environment_fragment_file']=str(ep);a['environment_fragment_sha256']=sha(ep)
    save(ap,a);save(scope/'semantic-equivalence.json',changes)
    return mp,ap,changes


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
        env=read(oldpre/'environment.json')
        env={k:v.replace(row['new_run'],str(RUN)) if isinstance(v,str) else v for k,v in env.items()}
        oldbootstrap=str(Path(read(p['scope_manifest'])['bootstrap_path']).parent)
        env['PYTHONPATH']=':'.join(x for x in env.get('PYTHONPATH','').split(':') if x and x!=oldbootstrap)
        assert not any(k in env for k in H.MASKS)
        save(pre/'environment.json',env)
        mp,ap,scope_changes=refreshed_scope(p)
        q=copy.deepcopy(p);q['scope_manifest']=str(mp);q['scope_activation']=str(ap);q['time']=H.now();q['runs']['gpu4'].update(new_run=str(RUN),config_changes=changes,recovery=recovery)
        q['runs']['gpu4']['prepared_sha256']={n:sha(pre/n) for n in ('original.yaml','resolved.yaml','environment.json')}
        evidence={str(f):sha(f) for f in (OLD/'plan.json',OLD/'rlt-stopped.json',D/'run/final.json',D/'run/cleanup.json',release,Path(__file__))}
        q['frozen_files'].update(evidence)
        q['frozen_files'].update({str(f):sha(f) for f in (mp,ap,mp.parent/'environment-fragment.json',mp.parent/'semantic-equivalence.json',mp.parent/'bootstrap/graphics_scope_runtime.py',mp.parent/'bootstrap/sitecustomize.py')})
        q['frozen_files'].update({path:item['sha256'] for path,item in scope_changes.items()})
        q['repair']={'reason':'ENAMETOOLONG before any driver dispatch','original_stage':str(OLD),'same_cycle_id_different_path':True,'no_new_stop':True,'scope_profile_names_only':scope_changes,'evidence':evidence}
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
