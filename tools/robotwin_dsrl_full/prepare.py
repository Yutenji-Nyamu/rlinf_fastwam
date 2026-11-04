"""Prepare from the actual Oct7 pi05 configs; no environment or GPU launch."""
import ast,copy,json,os,shutil,sys
from pathlib import Path
from omegaconf import OmegaConf
from tools.robotwin_dsrl_full.adapt import apply

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))

def main(root,directory):
    root,d=Path(root),Path(directory);d.mkdir(exist_ok=False)
    apply(root)
    oldroot=Path('/data/chenyiteng/results/rlinf-dsrl-pi05-u-20261006')
    plans=[]
    for role in ('clean','u'):
        old=oldroot/f'{role}-n8-w2000-fixed-formal200-1007-v1/runtime'
        c=OmegaConf.to_container(OmegaConf.load(old/'resolved.yaml'),resolve=True)
        before=copy.deepcopy(c);a=d/role;a.mkdir();(a/'config').mkdir()
        # Preserve the accepted instruction reset, physical GPU6 scope, model,
        # per-round sample count and optimizer settings, replacing only code/run.
        oldenv=json.loads((old/'environment.json').read_text())
        oldsource='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-pi05-u-sz1-20261006'
        oldscope='/data/chenyiteng/deployment-20261006/dsrl-u-lease-v1/scope-g7'
        scope='/data/chenyiteng/deployment-20261006/dsrl-u-lease-v1/scope-g6'
        env={k:str(v).replace(oldsource,str(root)).replace(oldscope,scope).replace('libdsrl_u_g7_20261006','libdsrl_u_g6_20261006') for k,v in oldenv.items()}
        env.update(RLINF_CODE_WORKING_DIR=str(root),REPO_PATH=str(root),EMBODIED_PATH=str(root/'examples/embodiment'),DOJO_GRPO_DEPLOYMENT=str(a),CUDA_VISIBLE_DEVICES='')
        c['cluster']['component_placement']={'actor,env,rollout':{'node_group':'dsrl_gpu6','placement':'6'}}
        group=c['cluster']['node_groups'][0];group['label']='dsrl_gpu6'
        worker_env={k:v for k,v in env.items() if k not in ('CUDA_VISIBLE_DEVICES','DOJO_GRPO_DEPLOYMENT')}
        group['env_configs'][0]['env_vars']=[{k:v} for k,v in sorted(worker_env.items())]
        assert Path(worker_env['LD_PRELOAD']).is_file()
        assert Path(worker_env['ROBOTWIN_PATH'],'robotwin/envs/vector_env.py').is_file()
        run=Path('/data/chenyiteng/results/robotwin-dsrl-full-20261104')/role
        model=c['actor']['model'];model['openpi'].update(dsrl_full_noise=True,dsrl_action_noise_dim=1600)
        model['openpi_data']['default_prompt']='turn on the switch'
        c['algorithm']['entropy_tuning'].update(initial_alpha=.02,target_entropy=-800)
        c['algorithm']['replay_buffer']['schema_version']=4 if role=='u' else 3
        for mode in ('train','eval'):
            e=c['env'][mode];e['seeds_path']=str(root/f'rlinf/envs/robotwin/seeds/turn_switch_{mode}_20260914.json')
            assert json.loads(Path(e['seeds_path']).read_text())['turn_switch']['success_seeds']
            e['task_config'].update(task_name='turn_switch',step_lim=200,save_path=str(run/f'robotwin_data/{mode}'),eval_video_log=False)
            e['video_cfg']['video_base_dir']=str(run/f'video/{mode}')
        c['runner'].update(max_steps=200,resume_dir=None,weight_sync_interval=1,overlap_env_bootstrap=False,per_worker_log=False,per_worker_log_path=None)
        c['runner']['logger'].update(log_path=str(run/'formal'),experiment_name='formal')
        c['actor']['fsdp_config']['checkpoint_format']='local_shard'
        assert c['actor']['global_batch_size']==c['actor']['micro_batch_size']==256
        assert c['actor']['model']['num_action_chunks']==20
        assert c['env']['train']['total_num_envs']==8 and c['algorithm']['replay_buffer']['warmup_size']==2000
        p=dict(training_source=str(root),run=str(run),python=group['env_configs'][0]['python_interpreter_path'],legacy_ops='/data/chenyiteng/results/server-maintenance-20260911/grpo-seed42-half/ops.py',namespace=f'RLinf_rt2_dsrl_full_{role}_20261104',gpu=6,model=model['model_path'],task='turn_switch',model_contract_required=False,protected_output=f'/data/chenyiteng/models/protected-eval/rt2-dsrl-full-20261104/{role}')
        write(a/'plan.json',p);write(a/'training-environment.json',env)
        OmegaConf.save(OmegaConf.create(c),a/'config/formal.yaml')
        for phase,steps in [('smoke',1),('smoke_resume',2)]:
            s=copy.deepcopy(c);s['runner'].update(max_steps=steps,val_check_interval=0,save_interval=1)
            s['runner']['logger'].update(log_path=str(run/'smoke'),experiment_name='smoke')
            s['env']['train']['max_steps_per_rollout_epoch']=40
            s['algorithm'].update(utd_ratio=1);s['algorithm']['replay_buffer']['warmup_size']=8
            if phase=='smoke_resume':s['runner']['resume_dir']=str(run/'smoke/smoke/checkpoints/global_step_1')
            OmegaConf.save(OmegaConf.create(s),a/f'config/{phase}.yaml')
        def flat(x,pre=''):
            if isinstance(x,dict):
                r={}
                for k,v in x.items():r.update(flat(v,pre+'.'+k if pre else k))
                return r
            return {pre:x}
        b,n=flat(before),flat(c)
        write(a/'config-diff.json',{k:dict(old=b.get(k),new=n.get(k)) for k in sorted(b.keys()|n.keys()) if b.get(k)!=n.get(k)})
        plans.append((role,a))
    phases=[]
    for role,a in plans:
        for phase in ['smoke','smoke_resume']:phases.append(dict(role=role,phase=phase,deployment=str(a),timeout=1800))
    for role,a in plans:phases.append(dict(role=role,phase='formal',deployment=str(a),timeout=604800))
    write(d/'plan.json',dict(gpu=6,gpus=[6],training_source=str(root),phases=phases,predecessors=[],budget='8 episodes x 200 rounds each; 200 actions; B256; UTD20; warmup2000; N10/H50/C20',failure='advance remaining method, then existing per-card RLT'))
    print(json.dumps(dict(prepared=str(d),formal_budget='200 rounds each, 8 episodes/round, action limit200',phases=phases)))

if __name__=='__main__':main(*sys.argv[1:])
