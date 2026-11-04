"""Derive Dojo+U and RT2+U from their two live Clean controls.

The U definition is the already accepted RT2 full-noise U implementation.
Only Dojo's executed prefix changes to C50, matching its native policy.
"""
import copy,json,shutil,sys,os
from pathlib import Path
from omegaconf import OmegaConf

B=Path('/data/chenyiteng/deployment-20261104')
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2))
def main(root):
    root=Path(root);dj=B/'dojo-dsrl-u-g4';rt=B/'rt2-dsrl-u-g7';pair=B/'parallel-u47'
    for d in (dj,rt,pair):d.mkdir(exist_ok=False)
    oldrt=B/'rt2-dsrl-full-g6-r3/u';olddj=B/'dojo-dsrl-full-g5'
    rules=read('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/plan.json')['slots']['4']
    u=OmegaConf.to_container(OmegaConf.load(oldrt/'config/formal.yaml'),resolve=True)
    for old,d,g,dojo in [(olddj,dj,4,True),(oldrt,rt,7,False)]:
        p=read(old/'plan.json');before=OmegaConf.to_container(OmegaConf.load(old/'config/formal.yaml'),resolve=True)
        run=Path('/data/chenyiteng/results/dojo-offpolicy/dojo-dsrl-u-g4-20261104' if dojo else '/data/chenyiteng/results/robotwin-dsrl-full-20261104/u-g7')
        changes={p['training_source']:str(root),str(old):str(d),p['run']:str(run)}
        if not dojo:changes.update({'scope-g6':'scope-g7','libdsrl_u_g6_':'libdsrl_u_g7_','dsrl_gpu6':'dsrl_gpu7'})
        def remap(x):
            text=json.dumps(x)
            for a,b in changes.items():text=text.replace(a,b)
            return json.loads(text)
        p,c=remap(p),remap(before)
        p.update(gpu=g,gpus=[g],namespace=f'RLinf_{d.name.replace("-","_")}_20261104',pre_registered=True,predecessors=[],model_contract_required=False,protected_output=f'/data/chenyiteng/models/protected-eval/{d.name}-20261104')
        p.pop('defer_formal_to',None)
        env=remap(read(old/'training-environment.json'));env['CUDA_VISIBLE_DEVICES']=''
        if dojo:
            p.update(render_gpus={'4':4},reviewed_graphics_residual_mib={},gpu_admission=rules)
            spec=copy.deepcopy(u['algorithm']['dsrl_u']);spec['spec']['chunk_length']=50
            c['algorithm']['dsrl_u']=spec;c['algorithm']['replay_buffer']['schema_version']=4
            for section in ('actor','rollout'):
                m=c[section]['model'];m['openpi']['dsrl_u_enabled']=True;m['openpi']['dsrl_u_spec']=copy.deepcopy(spec['spec'])
            for mode in ('train','eval'):c['env'][mode]['physical_gpus']=[g]
            nc=remap(read(old/'native-g5.json'));nc.update(gpu=g,output=str(d/'native-g4'),socket=str(d/'rpc/gpu4.sock'))
            (d/'rpc').mkdir();(d/'rpc/authkey').write_bytes(os.urandom(32));(d/'rpc/authkey').chmod(0o600)
            save(d/'native-g4.json',nc);save(d/'sim-environment.json',remap(read(old/'sim-environment.json')))
        if dojo:c['cluster']['component_placement']={'actor, env, rollout':str(g)}
        else:
            c['cluster']['component_placement']={'actor,env,rollout':{'node_group':f'dsrl_gpu{g}','placement':str(g)}}
            c['cluster']['node_groups'][0]['label']=f'dsrl_gpu{g}'
        # Keep each established backend's binding environment and point it at
        # this isolated source. RT2 retains its tested EGL physical-card shim.
        if not dojo:c['cluster']['node_groups'][0]['env_configs'][0]['env_vars']=[{k:v} for k,v in sorted(env.items()) if k not in ('CUDA_VISIBLE_DEVICES','DOJO_GRPO_DEPLOYMENT')]
        if not dojo:assert Path(env['LD_PRELOAD']).is_file()
        c['runner']['resume_dir']=None
        save(d/'training-environment.json',env)
        (d/'config').mkdir()
        for phase,steps in [('smoke',1),('smoke_resume',2),('formal',c['runner']['max_steps'])]:
            x=copy.deepcopy(c)
            if phase!='formal':
                x['runner'].update(max_steps=steps,val_check_interval=0,save_interval=1)
                x['runner']['logger'].update(log_path=str(run/'smoke'),experiment_name='smoke')
                x['env']['train']['max_steps_per_rollout_epoch']=50 if dojo else 40
                x['algorithm']['utd_ratio']=1;x['algorithm']['replay_buffer']['warmup_size']=8
                if phase=='smoke_resume':x['runner']['resume_dir']=str(run/'smoke/smoke/checkpoints/global_step_1')
            OmegaConf.save(OmegaConf.create(x),d/f'config/{phase}.yaml')
        commands=[dict(name=ph,command=[p['python'],'-u','-B',str(root/'tools/dojo_offpolicy/training_entry.py'),'--config-path',str(d/'config'),'--config-name',ph],timeout=1800 if ph!='formal' else None) for ph in ('smoke','smoke_resume','formal')]
        if dojo:p['phases']=commands
        save(d/'plan.json',p)
        save(d/'control-config.json',before);save(d/'formal-config.json',c)
        if not dojo:
            a=d/'owner';a.mkdir();save(a/'plan.json',dict(gpu=g,gpus=[g],training_source=str(root),predecessors=[],phases=[dict(role='u',phase=v['name'],deployment=str(d),timeout=v['timeout']) for v in commands]))
    save(pair/'prepared.json',dict(jobs=[str(dj),str(rt/'owner')],dojo=str(dj),rt2=str(rt),source=str(root),gpus=[4,7]))
    print(json.dumps(read(pair/'prepared.json')))

if __name__=='__main__':main(sys.argv[1])
