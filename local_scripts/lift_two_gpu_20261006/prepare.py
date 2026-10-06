"""Prepare a private two-GPU recipe from the currently proven lift recipe.

Preparation performs no process signals and no GPU calls. After the original
owner returns RLT, a separate lifecycle action borrows only cards 4 and 5.
"""
import argparse,ast,copy,hashlib,importlib.util,json,os,socket,subprocess,sys
from pathlib import Path

S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D=S/'lift-two-gpu-b32-v1'; O=S/'runs/lift-two-gpu-b32-v1'
OLD=S/'lift-pot-v1'; OLD_O=S/'runs/lift-pot-v1'
R=S/'rlinf-opendw-two-gpu-v1'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);assert not p.exists()
    p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    p.write_text(json.dumps(value,indent=2)+'\n');p.chmod(0o600)
def write(p,text):
    p=Path(p);assert not p.exists()
    p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    p.write_text(text);p.chmod(0o500 if p.suffix=='.py' else 0o600)
def replace(text,before,after,count=1):
    assert text.count(before)==count, ('Source anchor changed',before,text.count(before))
    return text.replace(before,after)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def command(args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True,timeout=180).strip()
def commit(args):
    publication=S/'publication/wmrl-bell-release-v1'
    name=command(['git','log','-1','--format=%an'],publication)
    email=command(['git','log','-1','--format=%ae'],publication)
    return command(['git','-c','user.name='+name,'-c','user.email='+email,*args],R)


def sources():
    assert not (D/'source-ready.json').exists() and not R.exists()
    old=read(OLD/'prepared/owner-plan.json'); donor=Path(old['repo'])
    assert command(['git','rev-parse','HEAD'],donor)==old['repo_head']
    # Copy the committed tree into an independent repository, then copy the
    # proven private overlay's exact changed and new source files.
    command(['git','clone','--shared','--no-checkout',str(donor),str(R)])
    command(['git','checkout','-b','codex/wmrl-two-gpu-b32-20261006',old['repo_head']],R)
    changed=command(['git','diff','HEAD','--name-only','--','rlinf','examples'],donor).splitlines()
    extra=command(['git','ls-files','--others','--exclude-standard','--','rlinf','examples'],donor).splitlines()
    for name in sorted(set(changed+extra)):
        source=donor/name;target=R/name
        assert source.is_file() and source.stat().st_size<5000000
        target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.is_symlink()
        target.write_bytes(source.read_bytes())
    # Keep only three narrow private implementation changes: checkpoint
    # re-sharding, policy batch measurements, and actor peak measurements.
    for source,name in [('resume_two_rank.py','wmrl_resume_two_rank.py'),('batch_probe.py','wmrl_batch_probe.py')]:
        target=R/'rlinf/utils'/name;assert not target.exists();target.write_bytes((D/'code'/source).read_bytes())
    manager=R/'rlinf/hybrid_engines/fsdp/fsdp_model_manager.py'
    text=manager.read_text();anchor='        self._strategy.load_checkpoint(\n'
    injected='''        if self._cfg.fsdp_config.get("resume_source_world_size", 1) == 2:
            from rlinf.utils.wmrl_resume_two_rank import load_two_rank
            load_two_rank(self.model, self.optimizer, self.lr_scheduler, load_path)
            return

'''
    manager.write_text(replace(text,anchor,injected+anchor))
    worker=R/'rlinf/workers/rollout/hf/huggingface_worker.py'
    text=worker.read_text();anchor='''        kwargs = (
            self._train_sampling_params'''
    injected='''        if mode == "train":
            from rlinf.utils.wmrl_batch_probe import run_policy_probe
            run_policy_probe(self, env_obs)
            if not getattr(self, "_wmrl_actual_batch_logged", False):
                self._wmrl_actual_batch_logged = True
                print("WMRL_POLICY_ACTUAL_BATCH " + str(env_obs["states"].shape[0]), flush=True)
'''
    worker.write_text(replace(text,anchor,injected+anchor))
    actor=R/'rlinf/workers/actor/embodied_fsdp_actor_worker.py'
    text=actor.read_text();tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EmbodiedFSDPActor')
    fn=next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='run_training')
    # Wrapper preserves the original function/decorators and records one peak
    # at the actual training boundary. No loss or optimizer math changes.
    lines=text.splitlines(keepends=True)
    start=fn.lineno-1
    assert lines[start].startswith('    def run_training(')
    lines[start]=lines[start].replace('def run_training(', 'def _wmrl_original_run_training(',1)
    text=''.join(lines)
    wrapper='''
    def run_training(self, *args, **kwargs):
        import json as _json
        import time as _time
        import math as _math
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _start = _time.perf_counter()
        result = self._wmrl_original_run_training(*args, **kwargs)
        torch.cuda.synchronize()
        _proof = {"seconds": _time.perf_counter()-_start,
                  "micro_batch_size": int(self.cfg.actor.micro_batch_size),
                  "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                  "reserved_peak_bytes": torch.cuda.max_memory_reserved()}
        if isinstance(result, dict):
            for _key, _value in result.items():
                if "grad_norm" in _key:
                    _norm = float(_value)
                    assert _math.isfinite(_norm)
                    _proof[_key] = _norm
        print("WMRL_ACTOR_MEMORY " + _json.dumps(_proof), flush=True)
        return result
'''
    # Class is the final top-level declaration in this exact donor file.
    assert cls.end_lineno==len(text.splitlines()) or not text.splitlines()[cls.end_lineno:]
    actor.write_text(text.rstrip()+'\n'+wrapper)
    for p in [manager,worker,actor,R/'rlinf/utils/wmrl_resume_two_rank.py',R/'rlinf/utils/wmrl_batch_probe.py']:
        ast.parse(p.read_text())
    command(['git','add','--','rlinf','examples'],R)
    commit(['commit','-m','Support single-rank WMRL resume and bounded batch measurements'])
    head=command(['git','rev-parse','HEAD'],R)
    generated=D/'generated';generated.mkdir(mode=0o700)
    source=Path(old['base_owner_module']);text=source.read_text()
    text=replace(text,'GPUS = [4, 5, 6, 7]','GPUS = [4, 5]')
    text=replace(text,"PLACEMENT = {'actor': '4,5', 'env': '6,7', 'rollout': '4,5'}","PLACEMENT = {'actor': '4', 'env': '5', 'rollout': '4'}")
    text=replace(text,"VISIBLE = {'actor': [['4'], ['5']], 'env': [['6'], ['7']], 'rollout': [['4'], ['5']]}","VISIBLE = {'actor': [['4']], 'env': [['5']], 'rollout': [['4']]}")
    text=replace(text,"assert [s['physical_gpu'] for s in services] == [6, 7]","assert [s['physical_gpu'] for s in services] == [5]")
    text=replace(text,"assert len({s['key'] for s in services}) == len({s['url'].rstrip('/') for s in services}) == 2","assert len({s['key'] for s in services}) == len({s['url'].rstrip('/') for s in services}) == 1")
    text=replace(text,"assert one_arg(argv, '--wm-batch-size') == '16'","assert one_arg(argv, '--wm-batch-size') == '32'")
    text=replace(text,'assert self.get_world_size(name) == 2','assert self.get_world_size(name) == 1')
    text=replace(text,"assert set(state['runs']) == {'gpu4', 'gpu5', 'gpu6', 'gpu7'}","assert set(state['runs']) == {'gpu4', 'gpu5'}")
    ast.parse(text);write(generated/'opendw_owner_base.py',text)
    for name in ['opendw_service_batched.py','wm_batch.py','opendw_action_telemetry.py','rm_adapter.py','rm_inference.py']:
        p=OLD/'generated/service'/name;text=p.read_text()
        if name=='wm_batch.py':text=replace(text,'MAX_WM_BATCH = 16','MAX_WM_BATCH = 32')
        if name=='opendw_service_batched.py':text=replace(text,'{("batched", 16), ("b1_reference", 1)}','{("batched", 16), ("batched", 32), ("b1_reference", 1)}')
        ast.parse(text);write(generated/'service'/name,text)
    # Generic parent validator handles the old combined four-card release.
    source=OLD/'prepared-cycles/lift-g567-v1/rlt_returned_cycle.py'
    for key,gpus in [('gpu4',[4]),('gpu567',[5])]:
        text=source.read_text()
        if key=='gpu567':
            text=replace(text,"expected = [4] if p['group'] == 'gpu4' else [5, 6, 7]","expected = [4] if p['group'] == 'gpu4' else [5]")
            text=replace(text,"B.GPUS = [4] if p['group'] == 'gpu4' else [5, 6, 7]","B.GPUS = [4] if p['group'] == 'gpu4' else [5]")
            anchor="    assert prior_plan['head'] == HEADS[group]\n"
            text=replace(text,anchor,anchor+"    assert group == 'gpu567'\n    prior_plan = copy.deepcopy(prior_plan)\n    prior_plan['runs'] = {'gpu5': prior_plan['runs']['gpu5']}\n")
        ast.parse(text);write(generated/key/'rlt_returned_cycle.py',text)
    combined=(OLD/'prepared-cycles/cycle/rlt_returned_multigpu_cycle.py').read_text()
    combined=combined.replace("CHILD_GPUS = {'gpu4': [4], 'gpu567': [5, 6, 7]}","CHILD_GPUS = {'gpu4': [4], 'gpu567': [5]}")
    combined=combined.replace('[4, 5, 6, 7]','[4, 5]').replace("{'gpu4', 'gpu5', 'gpu6', 'gpu7'}","{'gpu4', 'gpu5'}")
    ast.parse(combined);write(generated/'rlt_returned_multigpu_cycle.py',combined)
    save(D/'source-ready.json',{'repo':str(R),'head':head,'donor_head':old['repo_head'],
        'copied_donor_files':sorted(set(changed+extra)),'generated_sha256':{str(p):sha(p) for p in generated.rglob('*.py')}})


def lifecycle_and_recipe():
    assert not (D/'ready.json').exists()
    old=read(OLD/'prepared/owner-plan.json');final=read(OLD_O/'final.json')
    assert final['recovery_error'] is None and final['rlt_return_dispatched']
    scope=read(OLD/'prepared/graphics-fragment.json');manifest=Path(scope['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'])
    original=read(OLD/'prepared-cycles/cycle/plan.json')
    cycles=D/'cycles';cycles.mkdir(mode=0o700,exist_ok=True)
    children={}
    for key,name in [('gpu4','two-g4'),('gpu567','two-g5')]:
        source=D/'generated'/key/'rlt_returned_cycle.py'
        module=load('two_prepare_'+key,source)
        prior=original['children'][key];prior_plan=read(Path(prior['path'])/'plan.json')
        target=cycles/name
        module.prepare(target,Path(prior['path']),Path(prior['module']),OLD_O,Path(prior_plan['base_module']),
            scope_activation=old['graphics_scope']['activation_receipt'],scope_manifest=str(manifest),scope_id=prior_plan['scope_id'])
        children[key]={'path':str(target),'module':str(target/'rlt_returned_cycle.py')}
    combined=load('two_gpu_combined_prepare',D/'generated/rlt_returned_multigpu_cycle.py')
    combined.prepare(cycles/'cycle',children)
    checkpoints=sorted((OLD_O/'formal/lift-pot-v1-formal/checkpoints').glob('global_step_*'),key=lambda p:int(p.name.split('_')[-1]))
    resume=checkpoints[-1];step=int(resume.name.split('_')[-1]);assert step>=10 and step%10==0
    required=[resume/'actor/local_shard_checkpoint'/f'checkpoint_rank_{r}.pt' for r in (0,1)]
    assert all(p.stat().st_size>10000000000 for p in required)
    prepared=D/'prepared';prepared.mkdir(mode=0o700)
    donor=read(OLD/'prepared/formal.yaml')
    formal=copy.deepcopy(donor)
    formal['cluster']['component_placement']={'actor':'4','env':'5','rollout':'4'}
    formal['env']['train']['service_urls']=['http://127.0.0.1:18985']
    formal['actor']['micro_batch_size']=16
    formal['actor']['fsdp_config']['resume_source_world_size']=2
    formal['runner']['resume_dir']=str(resume)
    def route(cfg,key):
        cfg=json.loads(json.dumps(cfg).replace(str(OLD_O/'formal'),str(O/key)))
        for component in ('actor','env','rollout'):cfg[component]['group_name']='TwoGPU_'+key+'_'+component
        cfg['runner']['logger']['experiment_name']='lift-two-gpu-b32-'+key
        return cfg
    formal=route(formal,'formal')
    formal['runner']['resume_dir']=str(resume)
    smoke=copy.deepcopy(formal)
    smoke['runner'].update(max_steps=step+1,save_interval=1,val_check_interval=1)
    smoke['env']['train'].update(rollout_epoch=1,max_episode_steps=32,max_steps_per_rollout_epoch=32)
    smoke['env']['eval'].update(max_episode_steps=32,max_steps_per_rollout_epoch=32)
    smoke['env']['eval']['task_config']['step_lim']=32
    smoke['actor']['global_batch_size']=64
    smoke['rollout']['batch_probe_sizes']=[32,64,128]
    smoke=json.loads(json.dumps(smoke).replace(str(O/'formal'),str(O/'startup_smoke')))
    for component in ('actor','env','rollout'):smoke[component]['group_name']='TwoGPU_smoke_'+component
    smoke['runner']['logger']['experiment_name']='lift-two-gpu-b32-smoke'
    save(prepared/'formal.yaml',formal);save(prepared/'startup_smoke.yaml',smoke)
    environment=read(old['environment_file'])
    environment={k:str(v).replace(old['repo'],str(R)) for k,v in environment.items()}
    environment.update(RLINF_CODE_WORKING_DIR=str(R),HOME='/home/chenyiteng',USER='chenyiteng',LOGNAME='chenyiteng')
    save(prepared/'environment.json',environment)
    scope={k:str(v).replace(old['repo'],str(R)) for k,v in scope.items()}
    save(prepared/'graphics-fragment.json',scope)
    service=copy.deepcopy(old['services'][0]);service.update(key='wm5',physical_gpu=5,url=formal['env']['train']['service_urls'][0],cwd=str(D/'generated/service'))
    argv=service['argv'];argv[3]=str(D/'generated/service/opendw_service_batched.py')
    for flag,value in [('--physical-gpu','5'),('--port','18985'),('--wm-batch-size','32'),('--output-dir',str(O/'services/wm5/records'))]:argv[argv.index(flag)+1]=value
    service['environment']['PYTHONPATH']=service['environment']['PYTHONPATH'].replace(str(OLD/'generated/service'),str(D/'generated/service'))
    source_ready=read(D/'source-ready.json')
    plan={'mode':'two_gpu_b32_formal','physical_gpus':[4,5],'owner_dir':str(O),
        'lifecycle_path':str(cycles/'cycle'),'lifecycle_module':str(cycles/'cycle/rlt_returned_multigpu_cycle.py'),
        'python':old['python'],'repo':str(R),'repo_head':source_ready['head'],
        'base_owner_module':str(D/'generated/opendw_owner_base.py'),
        'environment_file':str(prepared/'environment.json'),'graphics_fragment':str(prepared/'graphics-fragment.json'),
        'scope_manifest_sha256':sha(manifest),'reference_config':str(OLD/'prepared/formal.yaml'),
        'resume_dir':str(resume),'resume_step':step,'restore_wait_seconds':60,'services':[service],
        'trials':[{'key':key,'config':str(prepared/(key+'.yaml')),'namespace':'opendw_two_gpu_'+key+'_1006',
                   'episode_steps':32 if key=='startup_smoke' else 384,'num_envs':64,
                   'timeout_seconds':3600 if key=='startup_smoke' else 60*86400} for key in ['startup_smoke','formal']]}
    files=list((D/'code').glob('*.py'))+list((D/'generated').rglob('*.py'))+list(cycles.rglob('*.py'))+list(cycles.rglob('plan.json'))+list(prepared.glob('*'))
    files += [manifest,OLD/'prepared/formal.yaml',Path(service['reward_checkpoint'])]
    for name in command(['git','ls-files','--','rlinf','examples'],R).splitlines():
        if name.endswith('.py') and (name in source_ready['copied_donor_files'] or name in ['rlinf/hybrid_engines/fsdp/fsdp_model_manager.py','rlinf/workers/rollout/hf/huggingface_worker.py','rlinf/workers/actor/embodied_fsdp_actor_worker.py','rlinf/utils/wmrl_batch_probe.py','rlinf/utils/wmrl_resume_two_rank.py']):files.append(R/name)
    plan['source_sha256']={str(p):sha(p) for p in files}
    save(prepared/'owner-plan.json',plan)
    wrapper=load('two_gpu_validate',D/'code/owner.py');module=wrapper.install(plan);module.validate(plan)
    save(D/'ready.json',{'plan':str(prepared/'owner-plan.json'),'plan_sha256':sha(prepared/'owner-plan.json'),
        'entrypoint':str(D/'code/owner.py'),'cpu_validate_passed':True,'resume_step':step,'physical_gpus':[4,5]})


if __name__=='__main__':
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('sources','recipe'));args=parser.parse_args()
    if args.action=='sources':sources()
    else:lifecycle_and_recipe()
