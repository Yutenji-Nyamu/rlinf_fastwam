"""Synchronous native RLinf/RoboTwin EXPO-FT. Isolated smoke and resume driver."""
from __future__ import annotations
import argparse, hashlib, io, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from rlinf.algorithms.expo_ft.backend import Pi05Backend,create_robotwin_env,clone_env_observation
from rlinf.algorithms.expo_ft.core import ExpoConfig,ExpoLearner
from rlinf.algorithms.expo_ft.replay import ChunkReplay

def log(run,event,**fields):
    row={'time':time.time(),'event':event,**fields}
    with (run/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def digest(value):
    h=hashlib.sha256()
    def feed(x):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().contiguous()
            h.update(str((tuple(x.shape),x.dtype)).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());feed(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:feed(v)
        else:h.update(repr(x).encode())
    feed(value);return h.hexdigest()

def check_finite(value):
    if isinstance(value,torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():raise FloatingPointError('Nonfinite checkpoint tensor')
    elif isinstance(value,dict):
        for v in value.values():check_finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:check_finite(v)

def demo_window(root,index=0):
    import pyarrow.parquet as pq
    paths=sorted(Path(root).glob('data/*/episode_*.parquet'))
    prepared=json.loads((Path(root)/'prepared.json').read_text())
    if not prepared.get('complete') or not paths:raise ValueError('No verified successful demonstration source')
    path=paths[index%len(paths)];table=pq.read_table(path)
    if table.num_rows<50:raise ValueError('Successful demo shorter than full real H50')
    start=min(20,table.num_rows-50)
    tasks_path=Path(root)/'meta'/'tasks.jsonl'
    tasks_bytes=tasks_path.read_bytes()
    tasks={}
    for line in tasks_bytes.decode('utf-8').splitlines():
        if not line.strip():continue
        item=json.loads(line);task_index=int(item['task_index'])
        if task_index in tasks:raise ValueError('Duplicate LeRobot task_index metadata')
        tasks[task_index]=item['task']
    task_indices=table['task_index'].slice(start,50).to_pylist()
    if len(set(task_indices))!=1:raise ValueError('FM H50 window crosses task descriptions')
    task_index=int(task_indices[0]);prompt=tasks[task_index]
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('No real demonstration task prompt')
    def picture(key):
        item=table[key][start].as_py()
        if not item.get('bytes'):raise ValueError('Demo image not embedded; explicit decoder needed')
        # Preserve the genuine RGB frame. The backend applies the same native
        # OpenPI data/model resize-and-pad transforms used for real observations.
        with Image.open(io.BytesIO(item['bytes'])) as image:
            return torch.from_numpy(np.array(image.convert('RGB'))).unsqueeze(0)
    obs={'main_images':picture('observation.images.cam_high'),
         'wrist_images':torch.stack([picture('observation.images.cam_left_wrist'),picture('observation.images.cam_right_wrist')],dim=1),
         'states':torch.tensor([table['observation.state'][start].as_py()],dtype=torch.float32),
         'task_descriptions':[prompt]}
    actions=torch.tensor(table['action'].slice(start,50).to_pylist(),dtype=torch.float32).unsqueeze(0)
    return obs,actions,{'path':str(path),'episode_frames':table.num_rows,'start':start,'end':start+50,
        'verified_source':prepared['dataset_revision'],'task_index':task_index,'prompt':prompt,
        'tasks_sha256':hashlib.sha256(tasks_bytes).hexdigest(),'rgb_frame_shape':list(obs['main_images'].shape[1:])}

def frozen_sample(backend):
    return digest({n:p.detach().reshape(-1)[:64] for n,p in backend.model.named_parameters() if n not in backend.trainable})

def save_checkpoint(run,backend,learner,replay,generator,episode,inputs_hash,online_success,success_windows):
    payload={'version':1,'inputs_hash':inputs_hash,'base':backend.state_dict(),'core':learner.state_dict(),
             'replay':replay.state_dict(),'candidate_rng':generator.get_state(),'episode':episode,
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
             'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
             'online_success':online_success,'success_windows':success_windows}
    check_finite(payload)
    hashes={k:digest(payload[k]) for k in ('base','core','replay','candidate_rng')}
    payload['hashes']=hashes
    path=run/f'checkpoint-{episode:04d}.pt'
    temp=path.with_suffix('.pt.partial');torch.save(payload,temp);os.replace(temp,path)
    receipt={'path':str(path),'bytes':path.stat().st_size,'hashes':hashes,'episode':episode,
             'core_updates':learner.update_calls,'base_updates':backend.base_updates,'finite':True}
    (run/'checkpoint.json').write_text(json.dumps(receipt,indent=2));return path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--resume');parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=4);parser.add_argument('--candidate-microbatch',type=int,default=1)
    args=parser.parse_args();run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    inputs_bytes=Path(args.inputs).read_bytes();inputs=json.loads(inputs_bytes);inputs_hash=hashlib.sha256(inputs_bytes).hexdigest()
    source=Path(__file__).resolve().parents[2]
    for name,expected in inputs['port_source_manifest'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Port source fingerprint differs: '+name)
    (run/'resolved.json').write_text(json.dumps({'inputs':inputs,'cli':vars(args),'inputs_sha256':inputs_hash},indent=2))
    seed=inputs['smoke']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    backend=Pi05Backend(OmegaConf.create(inputs['model']),device='cuda:0',lr=inputs['smoke']['base_lr'],
                        candidate_microbatch=args.candidate_microbatch,source_head=inputs['source_head'])
    generator=torch.Generator(device='cuda:0').manual_seed(seed)
    learner=ExpoLearner(ExpoConfig(**inputs['core']),device='cuda:0',seed=seed)
    replay=ChunkReplay(capacity=256,seed=seed)
    episode=0;online_success=0;success_windows=[]
    log(run,'initialized',base_contract=backend.contract,physical_gpu=inputs['physical_gpu'],core_config=inputs['core'])
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['version']!=1 or state['inputs_hash']!=inputs_hash:raise ValueError('Resume input fingerprint differs')
        check_finite(state)
        backend.load_state_dict(state['base']);learner.load_state_dict(state['core']);replay.load_state_dict(state['replay'])
        generator.set_state(state['candidate_rng'].cpu());episode=state['episode'];online_success=state['online_success'];success_windows=state['success_windows']
        restored={k:digest(v) for k,v in {'base':backend.state_dict(),'core':learner.state_dict(),'replay':replay.state_dict(),'candidate_rng':generator.get_state()}.items()}
        if restored!=state['hashes']:raise AssertionError(('Resume state hashes differ',restored,state['hashes']))
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        log(run,'resume_verified',checkpoint=args.resume,hashes=restored,core_updates=learner.update_calls,base_updates=backend.base_updates)
        del state
    frozen_before=frozen_sample(backend)
    env=create_robotwin_env(OmegaConf.create(inputs['env']),num_envs=1,seed_offset=0)
    log(run,'simulator_binding',**env.expo_renderer_binding)
    start=time.time();start_update=learner.update_calls
    try:
        for _ in range(args.episodes):
            if env.success_seeds is None:raise ValueError('Inherited control seed table is unavailable')
            env_seed=int(env.success_seeds[episode%env.success_seeds.numel()])
            obs,_info=env.reset(env_seeds=[env_seed]);frames=[];executed=[];chunks=0;rewards_total=0.;success=False
            log(run,'episode_started',episode=episode,env_seed=env_seed)
            done=False
            while not done:
                original=clone_env_observation(obs)
                base=backend.sample_normalized(obs,num_candidates=8,generator=generator)[:,:,:10,:14]
                selection=learner.select_actions(backend.critic_observation(obs),base)
                canonical=backend.decode(obs,selection['actions'])
                actual=[];discounted=0.;terminated=False;truncated=False
                for t in range(10):
                    frames.append(clone_env_observation(obs))
                    action=canonical[:,t:t+1,:].clone()
                    obs,reward,term,trunc,info=env.step(action,auto_reset=False)
                    # Replay stores the submitted canonical command. The native
                    # controller still applies its existing gripper/drive limits;
                    # this is not a measured low-level motor trajectory.
                    actual.append(action.cpu());executed.append(action[0,0].cpu())
                    r=float(torch.as_tensor(reward).reshape(-1)[0]);discounted+=(.99**t)*r;rewards_total+=r
                    terminated=bool(torch.as_tensor(term).any());truncated=bool(torch.as_tensor(trunc).any())
                    success=success or terminated;done=terminated or truncated
                    if done:break
                actions=torch.cat(actual,dim=1)
                normalized=backend.encode_executed(original,actions)[:,:,:14].detach().cpu()
                K=len(actual)
                if K<10:normalized=torch.cat([normalized,torch.zeros(1,10-K,14)],dim=1)
                replay.append({'env_obs':original,'next_env_obs':clone_env_observation(obs),'actions':normalized,
                    'canonical_executed_actions':actions,'reward':discounted,'continuation':0. if terminated else 1.,
                    'executed_steps':K,'terminated':terminated,'truncated':truncated,'episode':episode})
                chunks+=1
                log(run,'real_chunk',episode=episode,chunk=chunks,physical_steps=len(executed),executed_K=K,
                    reward=discounted,terminated=terminated,truncated=truncated,selected=int(selection['index'][0]),
                    candidate_count=selection['candidate_actions'].shape[1],selection_pair=selection['selection_q_indices'].tolist(),
                    proposal_distinct=bool((base[:,1:]-base[:,:1]).abs().max()>0))
            if success:
                online_success+=1;replay.mark_success(episode)
                for index in range(0,len(executed)-49,10):
                    success_windows.append((frames[index],torch.stack(executed[index:index+50]).unsqueeze(0)))
                success_windows=success_windows[-128:]
            log(run,'episode_finished',episode=episode,physical_steps=len(executed),chunks=chunks,success=success,reward=rewards_total)
            def next_candidates(next_obs):
                return backend.sample_normalized(next_obs['env_obs'],num_candidates=8,generator=generator)[:,:,:10,:14]
            def fm_callback():
                if success_windows:
                    fm_obs,fm_actions=success_windows[-1];fm_source={'kind':'online_success_episode'}
                else:
                    fm_obs,fm_actions,evidence=demo_window(inputs['demo_path'],episode);fm_source={'kind':'verified_success_demo',**evidence}
                log(run,'base_fm_source',**fm_source)
                return backend.fm_update(fm_obs,fm_actions)
            before_core={name:digest(module.state_dict()) for name,module in [('critic',learner.critic),('editor',learner.editor),('target',learner.target_critic)]}
            before_temp=float(learner.temperature.detach())
            metrics=learner.update_call(lambda:replay.sample(args.batch_size,backend,'cuda:0'),next_candidates,fm_callback)
            if metrics['base/base_sampled_parameter_delta_max']<=0:raise AssertionError('No actual VLA FM parameter change')
            changed={name:digest(module.state_dict())!=before_core[name] for name,module in [('critic',learner.critic),('editor',learner.editor),('target',learner.target_critic)]}
            changed['temperature']=float(learner.temperature.detach())!=before_temp
            if not all(changed.values()):raise AssertionError(('No actual component update',changed))
            if frozen_sample(backend)!=frozen_before:raise AssertionError('Frozen VLM prefix changed')
            log(run,'real_update',metrics=metrics,parameter_changes=changed,base_prefix_samples_unchanged=True)
            episode+=1
            cp=save_checkpoint(run,backend,learner,replay,generator,episode,inputs_hash,online_success,success_windows)
            log(run,'checkpoint_saved',path=str(cp),update_calls=learner.update_calls)
    finally:
        env.offload()
    result={'ok':True,'completed_episodes':args.episodes,'total_episode':episode,'real_update_calls':learner.update_calls-start_update,
            'core_update_calls':learner.update_calls,'base_updates':backend.base_updates,'online_success_episodes':online_success,
            'elapsed_seconds':time.time()-start,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'env_closed':True,
            'checkpoint':str(cp),'resume_verified':bool(args.resume)}
    (run/'complete.json').write_text(json.dumps(result,indent=2));log(run,'complete',**result)

if __name__=='__main__':main()
