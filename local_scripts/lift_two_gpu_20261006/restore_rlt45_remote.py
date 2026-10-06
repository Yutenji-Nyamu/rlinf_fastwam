"""Restore only stopped RLT4/5 through fresh, audited return contracts."""
import copy,datetime,fcntl,hashlib,importlib.util,json,os,socket,sys
from pathlib import Path
from omegaconf import OmegaConf

assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
OLD=S/'lift-two-gpu-from0-v2';OWNER=S/'runs/lift-two-gpu-from0-v2'
F=S/'rlt45-return-20261007-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,value):
    p=Path(p);assert not p.exists()
    p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    p.write_text(json.dumps(value,indent=2)+'\n');p.chmod(0o600)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def main():
    assert not F.exists()
    prior=read(OLD/'prepared/owner-plan.json');oldcycle=Path(prior['lifecycle_path'])
    previous=load('prior_rlt45',Path(prior['lifecycle_module']))
    previous.install_helper(oldcycle);H=previous.H;cp=previous.load_plan(oldcycle)
    final=read(OWNER/'final.json');release=read(OWNER/'smoke-release.json')
    assert final['terminal_status']=='failed' and final['rlt_borrowed'] and not final['rlt_return_dispatched']
    assert final['recovery_error']['type']=='RuntimeError'
    assert 'New NVIDIA device rule requires precedence audit:' in final['recovery_error']['error']
    assert not H.same(read(OWNER/'owner-identity.json'))
    assert read(OWNER/'cleanup.json')['all_stopped'] and release['gpus']==[4,5]
    assert all(not H.same(x) for x in release['managed_processes'])
    assert not H.gpu_processes([4,5])
    actors=H.actors(cp)
    for key,row in cp['runs'].items():
        assert not H.same(row['original_identity'])
        assert not H.active(actors,row['original_namespace']) and not H.active(actors,row['namespace'])
        rt=Path(row['new_run'])/'runtime'
        assert not (rt/'driver-identity.json').exists() and not (rt/'driver.log').exists()
        # Failure was before Popen; reject any unattributed CPU driver too.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                if proc.stat().st_uid!=20001:continue
                argv=(proc/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            assert os.fsencode(row['new_run']) not in argv and os.fsencode(str(rt/'resolved.yaml')) not in argv
    F.mkdir(mode=0o700)
    # Added profiles select only explicit Norm preload libraries. Preserve them;
    # pin the audited files in a new private scope manifest for RLT4/5.
    oldchild=read(Path(cp['children']['gpu4']['path'])/'plan.json')
    manifest=read(oldchild['scope_manifest']);activation=read(oldchild['scope_activation'])
    audited={}
    for gpu in [6,7]:
        path=Path('/home/chenyiteng/.nv/nvidia-application-profiles-rc.d')/f'00-norm-g{gpu}-20261007.json'
        expected={'rules':[{'pattern':{'feature':'dso','matches':f'libnorm_g{gpu}_20261007.so'},
                             'profile':['EGLVisibleDGPUDevices',1<<gpu]}]}
        assert read(path)==expected and path.stat().st_uid==20001
        record={'sha256':sha(path),'size':path.stat().st_size}
        manifest['existing_profiles'][str(path)]=record;audited[str(path)]=record
    manifest['receipts_dir']=str(F/'scope/receipts');(F/'scope/receipts').mkdir(parents=True,mode=0o700)
    mp=F/'scope/scope.json';save(mp,manifest)
    runtime=load('audited_scope',manifest['runtime_path']);runtime.read_manifest(mp)
    activation.update(manifest=str(mp),manifest_sha256=sha(mp),activation_receipt=str(F/'scope/activation.json'))
    activation['environment_fragment']=copy.deepcopy(activation['environment_fragment'])
    activation['environment_fragment']['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST']=str(mp)
    activation['environment_fragment_file']=str(F/'scope/environment-fragment.json')
    save(F/'scope/environment-fragment.json',activation['environment_fragment'])
    activation['environment_fragment_sha256']=sha(F/'scope/environment-fragment.json')
    activation['derived_from']={'path':oldchild['scope_activation'],'sha256':sha(oldchild['scope_activation']),
                                'reason':'Audit disjoint Norm6/7 DSO profiles; no profile or mask changes'}
    save(F/'scope/activation.json',activation)
    children={};stops={}
    for group,childinfo in cp['children'].items():
        src=Path(childinfo['path']);p=copy.deepcopy(read(src/'plan.json'));stop=read(src/'rlt-stopped.json')
        assert not (src/'resumed-dispatched.json').exists() and not list(src.glob('*-launched.json'))
        target=F/('return-g4' if group=='gpu4' else 'return-g5');target.mkdir(mode=0o700)
        component=previous.CHILDREN[group];helper=component.H
        # Retain the proven sources and checkpoint contract. All new output
        # paths and namespaces belong exclusively to this return attempt.
        for source in src.glob('*.py'):
            dest=target/source.name;dest.write_bytes(source.read_bytes());dest.chmod(0o500)
        p['cycle_id']=target.name;p['time']=H.now()
        for field in ['base_module','helper_path','next_six_ops']:
            p[field]=str(target/Path(p[field]).name)
        p['scope_manifest']=str(mp);p['scope_activation']=str(F/'scope/activation.json')
        p['old_owner']=read(OWNER/'owner-identity.json')
        p['completed_owner']=str(OWNER);p['previous_cycle']=str(src)
        p['management_namespace']='opendw-rlt45-return-'+target.name
        p['frozen_files'].update({str(x):sha(x) for x in target.glob('*.py')})
        p['frozen_files'].update({str(x):sha(x) for x in [mp,F/'scope/activation.json',src/'plan.json',
                                                       src/'rlt-stopped.json',OWNER/'final.json',OWNER/'smoke-release.json']})
        for key,row in p['runs'].items():
            previous_new=Path(row['new_run']);new_run=previous_new.with_name(previous_new.name+'-audit1007')
            namespace='rlt-opendw-audit1007-'+key
            assert not new_run.exists() and not H.active(actors,namespace)
            config=helper.config(src/'prepared'/key/'resolved.yaml')
            recovery=stop['runs'][key]['recovery'];checkpoint=recovery['checkpoint']
            assert recovery['mode']=='resume_checkpoint' and checkpoint
            newcfg,changes=helper.resumed_config(config,previous_new,new_run,checkpoint['path'])
            env=read(src/'prepared'/key/'environment.json')
            env={k:v.replace(str(previous_new),str(new_run)).replace(row['namespace'],namespace) for k,v in env.items()}
            assert not any(k in env for k in helper.MASKS)
            assert 'libnorm_' not in env.get('LD_PRELOAD','')
            pre=target/'prepared'/key;pre.mkdir(parents=True,mode=0o700)
            (pre/'original.yaml').write_bytes((src/'prepared'/key/'original.yaml').read_bytes())
            OmegaConf.save(OmegaConf.create(newcfg),pre/'resolved.yaml',resolve=True)
            save(pre/'environment.json',env)
            row.update(new_run=str(new_run),namespace=namespace,recovery=recovery,
                       config_changes=changes,dependencies=helper.dependency_snapshot(newcfg),
                       prepared_sha256={n:sha(pre/n) for n in ['original.yaml','resolved.yaml','environment.json']})
            stop['runs'][key]['resolved_sha256']=sha(pre/'resolved.yaml')
        save(target/'plan.json',p)
        newcomponent=load('new_return_'+group,target/'rlt_returned_cycle.py')
        newcomponent.install_helper(target);newcomponent.load_plan(target)
        # This exercises the new scope overlay on the exact return environment.
        for key in p['runs']:
            effective=newcomponent.H.read(target/'prepared'/key/'environment.json')
            assert effective['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST']==str(mp)
            assert 'libnorm_' not in effective.get('LD_PRELOAD','')
        stop['cycle_id']=target.name;stop['time']=H.now()
        stops[group]=stop;children[group]={'path':str(target),'module':str(target/'rlt_returned_cycle.py')}
    combined=load('new_return_combined_source',prior['lifecycle_module'])
    cycle=F/'cycle';combined.prepare(cycle,children)
    for group,row in children.items():save(Path(row['path'])/'rlt-stopped.json',stops[group])
    combined=load('new_return_combined',cycle/'rlt_returned_multigpu_cycle.py')
    combined.install_helper(cycle);combined.finalize_stopped(cycle)
    new_release=copy.deepcopy(release);new_release['cycle_id']=cycle.name
    save(F/'release.json',new_release)
    save(F/'audit.json',{'time':H.now(),'physical_gpus':[4,5],'wm_deferred_by_user':True,
        'parent_owner':str(OWNER),'parent_cycle':str(oldcycle),'profiles_audited_unchanged':audited,
        'no_return_driver_launched':True,'checkpoint_steps':{k:r['recovery']['checkpoint']['step'] for k,r in cp['runs'].items()}})
    save(oldcycle/'superseded-by-rlt45-audit1007.json',{'time':H.now(),'new_cycle':str(cycle),'reason':'User requested RLT; prior return failed before Popen'})
    with (cycle/'operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=combined.resume(cycle,F/'release.json')
    save(F/'dispatched.json',{'time':H.now(),'result':result,'cycle':str(cycle),'physical_gpus':[4,5]})
    print(json.dumps(read(F/'dispatched.json')),flush=True)


if __name__=='__main__':
    with (OLD/'cycles/cycle/operation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
