"""One fresh reborrow plan: short formal-scale startup check, then formal run."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
D=S/'formal-control-v2'
O=S/'runs/formal-v2'
P=S/'runs/formal-v1'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def proc(path):
    tail=(path/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=int(path.name),uid=path.stat().st_uid,start=int(tail[19]),ppid=int(tail[1]),comm=(path/'comm').read_text().strip(),cmdline_sha256=sha(path/'cmdline'))

def main():
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    code=D/'formal';state=D/'prepared'
    assert not state.exists() and not O.exists()
    ready=read(D/'ready.json');assert ready['all_cpu_tests_passed']
    for name,h in ready['source_sha256'].items():assert sha(name)==h
    old=read(P/'owner-plan.json');final=read(P/'final.json')
    assert final['terminal_status']=='failed' and final['recovery_error'] is None and final['rlt_return_dispatched']
    sys.path.insert(0,str(code))
    G=load('repair_scope_prepare',code/'graphics_scope_prepare.py')
    R=load('repair_child_cycle',code/'rlt_returned_cycle.py')
    C=load('repair_combined_cycle',code/'rlt_returned_multigpu_cycle.py')
    B=load('repair_config_builder',code/'build_formal_config.py')
    state.mkdir(mode=0o700)
    # Only pre-existing system login helpers are explicitly reviewed exceptions.
    audits=[];parents=[];gpu_pids=G.gpu_process_pids()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid!=20001:continue
            tail=(path/'stat').read_text().rsplit(')',1)[1].split()
            if tail[0] in ('Z','X'):continue
            try:(path/'environ').read_bytes()
            except PermissionError:
                row=proc(path);assert row['comm'] in ('(sd-pam)','sshd')
                parent=Path('/proc')/str(row['ppid']);parent_row=proc(parent)
                cmd=(path/'cmdline').read_bytes();parent_cmd=(parent/'cmdline').read_bytes()
                if row['comm']=='(sd-pam)':
                    assert row['pid']==1606714 and row['start']==637903889
                    assert parent_row['comm']=='systemd' and parent_row['uid']==20001 and b'--user' in parent_cmd
                else:
                    assert cmd.startswith(b'sshd: chenyiteng') and parent_row['comm']=='sshd' and parent_row['uid']==0
                G.verify_unreadable_cpu_process(path,row,gpu_pids)
                audits.append(row);parents.append(parent_row)
        except FileNotFoundError:continue
    save(state/'system-process-audit.json',dict(exact_exceptions=audits,parents=parents,gpu_context_pids=sorted(gpu_pids)))
    # CPU check of the previously failing scan while RLT is still intact.
    scopes=G.scoped_processes(unreadable_cpu_exemptions=audits)
    save(state/'scope-scan-preflight.json',dict(permission_scan_passed=True,existing_rlt_scopes=scopes))
    prior_cycle=read(Path(old['lifecycle_path'])/'plan.json')
    prior4=read(Path(prior_cycle['children']['gpu4']['path'])/'plan.json')
    env4=read(Path(prior4['runs']['gpu4']['new_run'])/'runtime/environment.json')
    legacy=read(env4['RLINF_OPENDW_GPU_SCOPE_MANIFEST'])
    assert sha(legacy['profile_path'])==legacy['profile_sha256']
    base_env=read(old['environment_file'])
    scope_dir=(state/'graphics-scope').resolve();scope_id='opendwformal20261004v2'
    staged=G.prepare_stage(scope_dir,scope_id,[legacy['profile_path']],inherited=base_env)
    cycle=state/'cycle';children={}
    base=S/'multigpu-control-v1/multigpu/rlt_gpu567_cycle.py'
    for key in ('gpu4','gpu567'):
        prior=prior_cycle['children'][key];child=state/('reborrow-'+key+'-v2')
        R.prepare(child,Path(prior['path']),Path(prior['module']),P,base,
            scope_activation=cycle/'scope-activation.json',scope_manifest=scope_dir/'scope.json',scope_id=scope_id)
        children[key]=dict(path=str(child),module=str(child/'rlt_returned_cycle.py'))
    C.prepare(cycle,children)
    services=copy.deepcopy(old['services'])
    for row in services:
        argv=row['argv'];argv[argv.index('--output-dir')+1]=str(O/'services'/row['key']/'records')
    seeds=state/'native-seeds.json'
    oldcfg=read(old['trials'][0]['config'])
    save(seeds,read(oldcfg['env']['eval']['seeds_path']))
    reference=read(old['protocol_reference']['config'])
    control=read(S/'formal-control-v1/direct-state/control-config.json')
    formal,contract=B.build(reference,control,name='opendw-adjust-bottle-formal-v2',run_dir=str(O/'formal/run'),
        services=[r['url'] for r in services],native_assets=oldcfg['env']['eval']['assets_path'],native_eval_seeds=str(seeds))
    smoke=copy.deepcopy(formal)
    smoke['runner'].update(max_epochs=1,max_steps=1,save_interval=1,val_check_interval=1)
    smoke['env']['train'].update(rollout_epoch=1,max_episode_steps=32,max_steps_per_rollout_epoch=32)
    smoke['actor']['global_batch_size']=64
    smoke['env']['eval'].update(max_episode_steps=32,max_steps_per_rollout_epoch=32)
    smoke['env']['eval']['task_config']['step_lim']=32
    smoke=json.loads(json.dumps(smoke).replace(str(O/'formal'),str(O/'startup_smoke')).replace('opendw-adjust-bottle-formal-v2','opendw-adjust-bottle-startup-v2'))
    trial_rows=[]
    for key,cfg,length,timeout in [('startup_smoke',smoke,32,3600),('formal',formal,384,60*86400)]:
        cfg_path=state/(key+'.yaml');save(cfg_path,cfg)
        trial_rows.append(dict(key=key,num_envs=64,episode_steps=length,config=str(cfg_path),config_sha256=sha(cfg_path),namespace='opendw_sz3_'+key+'_v2',timeout_seconds=timeout))
    env_path=state/'environment.json';save(env_path,base_env);env_path.chmod(0o600)
    source=dict(old['source_sha256']);source.update(ready['source_sha256'])
    source[str(cycle/'rlt_returned_multigpu_cycle.py')]=sha(cycle/'rlt_returned_multigpu_cycle.py')
    for row in children.values():
        child=Path(row['path']);cp=read(child/'plan.json')
        source.update(cp['frozen_files']);source[str(child/'plan.json')]=sha(child/'plan.json')
    plan=dict(mode='multigpu_formal',start_mode='direct_start_user_override_20261004',startup_smoke=True,
        owner_dir=str(O),lifecycle_path=str(cycle),lifecycle_module=str(cycle/'rlt_returned_multigpu_cycle.py'),
        base_owner_module=old['base_owner_module'],python=old['python'],repo=old['repo'],repo_head=old['repo_head'],
        physical_gpus=[4,5,6,7],environment_file=str(env_path),source_sha256=source,services=services,trials=trial_rows,
        restore_wait_seconds=60,native_eval_seeds_sha256=sha(seeds),protocol_reference=old['protocol_reference'],
        handoff_evidence=dict(owner_final_path=str(P/'final.json'),owner_final_sha256=sha(P/'final.json')),
        graphics_scope=dict(scope_dir=str(scope_dir),environment_fragment_file=staged['environment_fragment_file'],
            environment_fragment_sha256=staged['environment_fragment_sha256'],post_borrow_hook_module=str(code/'post_borrow_hook.py'),
            prepare_module=str(code/'graphics_scope_prepare.py'),activation_receipt=str(cycle/'scope-activation.json'),
            audited_unreadable_cpu_processes=audits),budget=dict(effective_runner_iterations=200,N=64,G=8,R=8,C=32,L=384,
            startup_smoke=dict(N=64,G=8,R=1,L=32,global_batch=64,native_eval_N=32,native_eval_L=32),save_interval=10,native_eval_interval=10))
    save(state/'plan.json',plan);save(state/'formal-contract.json',contract)
    F=load('repair_formal_preflight',code/'opendw_formal_owner.py');m=F.install(plan);m.validate(plan)
    save(D/'prepared.json',dict(plan=str(state/'plan.json'),plan_sha256=sha(state/'plan.json'),owner=str(O),
        current_rlt_preserved=True,cpu_preflight_passed=True,system_process_exceptions=audits,
        checkpoint_steps={key:{k:r['recovery']['checkpoint']['step'] for k,r in read(Path(row['path'])/'plan.json')['runs'].items()} for key,row in children.items()}))
    print(json.dumps(read(D/'prepared.json')),flush=True)

if __name__=='__main__':main()
