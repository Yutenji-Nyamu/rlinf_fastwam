"""Run original checkpoint inference on matched RoboTwin evaluation scenes.

No optimizer, rollout scheduling, chunk conversion or teacher policy is replaced.
The native simulator is discarded between batches to release renderer resources.
"""
import argparse, dataclasses, json, os, random, resource, time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from tools.teacher_eval.protocol import episode_schedule, batches, summarize, paired_comparison


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def build_models(cfg, kind, checkpoint):
    from rlinf.models import get_model
    if kind=='dsrl':
        mc=OmegaConf.create(OmegaConf.to_container(cfg.rollout.model,resolve=True))
        mc.load_to_device=False
        # No critic forward is used by this evaluator.
        mc.openpi.dsrl_compile_q=False
        model=get_model(mc)
        weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not isinstance(weights,dict): raise TypeError('Expected exported policy state_dict')
        actual=model.state_dict()
        for k,v in weights.items():
            if k not in actual or actual[k].shape!=v.shape:
                raise ValueError(f'Policy tensor mismatch: {k}')
        report=model.load_state_dict(weights,strict=False)
        # The exporter records names; reject a silently skipped policy tensor.
        missing=[k for k in report.missing_keys if k.startswith(('dsrl_action_noise_net.','actor_image_encoder.','actor_state_encoder.'))]
        if missing or report.unexpected_keys: raise ValueError((missing,report.unexpected_keys))
        if int(model.dsrl_policy_phase.item())!=1: raise ValueError('DSRL checkpoint is still Gaussian warmup')
        model.requires_grad_(False).eval()
        return dict(model=model, loaded_keys=list(weights), chunk_steps=int(mc.num_action_chunks))
    if kind=='rlt':
        pc=OmegaConf.create(OmegaConf.to_container(cfg.rollout.model,resolve=True)); pc.load_to_device=False
        model=get_model(pc)
        weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(weights,strict=True)
        fc=OmegaConf.create(OmegaConf.to_container(cfg.rollout.rlt_feature_model,resolve=True)); fc.load_to_device=False
        feature=get_model(fc)
        from rlinf.algorithms.rlt.route import build_rlt_route
        route=build_rlt_route(cfg)
        if cfg.algorithm.rlt_route.type!='full_task': raise ValueError('This baseline requires the original full-task RLT checkpoint')
        model.requires_grad_(False).eval(); feature.requires_grad_(False).eval()
        return dict(model=model, feature=feature, route=route, loaded_keys=list(weights), chunk_steps=int(pc.num_action_chunks))
    raise ValueError(kind)


def predict(models, kind, condition, obs):
    obs=dict(obs); obs.setdefault('extra_view_images',None)
    if kind=='dsrl':
        model=models['model']; model.config=dataclasses.replace(model.config,use_dsrl=(condition=='teacher'))
        actions,result=model.predict_action_batch(obs,mode='eval',compute_values=False)
        return actions, dict(learned_noise=condition=='teacher', phase=int(model.dsrl_policy_phase.item()))
    feature=models['feature']
    if condition=='base':
        if feature.config.rlt_action_adapter=='identity':
            actions=feature.extract_rlt_obs(obs)['ref_chunk']
        else:
            features,context=feature.extract_rlt_obs(obs,return_decode_context=True)
            actions=feature.decode_rlt_action(features['ref_chunk'],context)
        return actions,dict(actor_switch=False)
    from rlinf.algorithms.rlt.rollout import predict_rlt_actions
    actions,result=predict_rlt_actions(policy_model=models['model'],feature_model=feature,
            rlt_route=models['route'],env_obs=obs,final_obs=None,mode='eval')
    flags=result['forward_inputs']['actor_switch']
    if not bool(flags.all()): raise ValueError('Teacher route did not execute the RLT actor')
    return actions,dict(actor_switch=True)


def environment(cfg, seeds):
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.envs.robotwin.process_vector_env import ProcessVectorEnv
    class FixedSeedEnv(RoboTwinEnv):
        def _init_reset_state_ids(self):
            self.reset_state_ids=torch.tensor(seeds,dtype=torch.long)
        def _init_env(self):
            os.environ['ASSETS_PATH']=self.cfg.assets_path
            self.venv=ProcessVectorEnv(factory='tools.teacher_eval.vector_probe:SnapshotVectorEnv',
                 torch_threads=1, task_config=OmegaConf.to_container(self.cfg.task_config,resolve=True),
                 n_envs=self.num_envs,env_seeds=seeds)
    ec=OmegaConf.create(OmegaConf.to_container(cfg.env.eval,resolve=True))
    ec.auto_reset=False; ec.total_num_envs=len(seeds); ec.group_size=1
    ec.video_cfg.save_video=False
    return FixedSeedEnv(ec,len(seeds),0,1,None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--validate-only',action='store_true')
    ap.add_argument('--parallel',type=int);args=ap.parse_args()
    job=json.loads(Path(args.job).read_text());out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    if args.parallel:job['parallel']=args.parallel
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(max(soft,8192),hard),hard))
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cfg=OmegaConf.load(job['config']); models=build_models(cfg,job['kind'],job['checkpoint'])
    chunk=models['chunk_steps']; max_steps=int(cfg.env.eval.max_episode_steps)
    assert chunk==int(job['chunk_steps']) and max_steps==200
    schedule=episode_schedule(job['seeds'],job['repeats']); all_batches=batches(schedule,job['parallel'])
    write(out/'contract.json',dict(kind=job['kind'],chunk_steps=chunk,max_steps=max_steps,
         episodes=len(schedule),distinct_scenes=len(job['seeds']),repeats=job['repeats'],parallel=job['parallel'],
         checkpoint=job['checkpoint'],config=job['config'],loaded_keys=models['loaded_keys'],
         eval_deterministic=bool(cfg.rollout.model.get('openpi',{}).get('dsrl_eval_deterministic',True)),
         time=time.time()))
    if args.validate_only:
        print('CPU model/checkpoint validation passed',job['kind'],chunk,flush=True);return
    for k in ['model','feature']:
        if k in models: models[k].to('cuda')
    started=time.monotonic(); completed={}
    for condition in ['base','teacher']:
        rows=[]
        for batch_id,items in enumerate(all_batches):
            path=out/condition/f'batch-{batch_id}.json'
            if path.exists():
                saved=json.loads(path.read_text())
                if [r['episode_id']for r in saved]!=[i['episode_id']for i in items]: raise ValueError('Existing batch schedule differs')
                rows+=saved; continue
            write(out/'progress.json',dict(condition=condition,batch=batch_id,stage='reset',completed=len(rows),time=time.time()))
            env=environment(cfg,[e['scene_seed']for e in items]); history=[]
            try:
                obs,_=env.reset(); initial=env.venv._call('episode_state')
                write(out/f'{condition}-batch-{batch_id}-initial.json',[{**item,**initial[i]}for i,item in enumerate(items)])
                np.savez_compressed(out/f'{condition}-batch-{batch_id}-initial.npz',states=obs['states'].numpy(),
                    main_images=obs['main_images'].numpy(),scene_seeds=np.array([e['scene_seed']for e in items]))
                if condition=='base' and batch_id==0:
                    with torch.random.fork_rng(devices=[0]), torch.inference_mode():
                        probe,proof=predict(models,job['kind'],'teacher',obs)
                    assert tuple(probe.shape)==(len(items),chunk,14) and bool(torch.isfinite(probe).all())
                    write(out/'teacher-inference-check.json',dict(time=time.time(),shape=list(probe.shape),**proof))
                    del probe
                success=np.zeros(len(items),dtype=bool);first_success=[None]*len(items)
                queries=max_steps//chunk
                for query in range(queries):
                    # Independent repeated rollouts, identical RNG schedule in each pair.
                    rng=int(job['noise_seed'])+batch_id*10007+query*101
                    random.seed(rng);np.random.seed(rng);torch.manual_seed(rng);torch.cuda.manual_seed_all(rng)
                    begin=time.monotonic()
                    with torch.inference_mode(): actions,proof=predict(models,job['kind'],condition,obs)
                    if tuple(actions.shape)!=(len(items),chunk,14):raise ValueError(f'Unexpected action shape {actions.shape}')
                    if not torch.isfinite(actions).all():raise ValueError('Non-finite action')
                    torch.cuda.synchronize();inference=time.monotonic()-begin
                    begin=time.monotonic();result=env.chunk_step(actions);obs=result[0][-1]
                    meta=env.venv._call('episode_state');native_success=np.array([x['success']for x in meta])
                    observed=env.success_once.cpu().numpy().astype(bool);current=np.logical_or(native_success,observed)
                    for i in np.flatnonzero(current & ~success): first_success[i]=(query+1)*chunk
                    success|=current
                    tick=dict(query_index=query,inference_s=inference,environment_s=time.monotonic()-begin,
                              successes=int(success.sum()),success_flags=success.tolist(),teacher_proof=proof,
                              model_peak_gib=torch.cuda.max_memory_allocated()/2**30,
                              native_action_count=[x['native_action_count']for x in meta])
                    history.append(tick)
                    write(out/'progress.json',dict(condition=condition,batch=batch_id,stage='rollout',
                          query=query+1,queries=queries,completed=len(rows),time=time.time(),**tick))
                    print(condition,batch_id,query+1,'success',int(success.sum()),'/',len(items),flush=True)
                batch_rows=[{**item,**initial[i],'success':bool(success[i]),'first_success_step':first_success[i],
                             'native_action_count':meta[i]['native_action_count'],'chunk_steps':chunk,
                             'submitted_steps':max_steps,'condition':condition}for i,item in enumerate(items)]
                write(path,batch_rows);write(path.with_name(path.stem+'-timing.json'),history)
                rows+=batch_rows;write(out/f'{condition}.json',rows)
            finally: env.offload(clear_cache=True)
        completed[condition]=rows
        write(out/f'{condition}-summary.json',summarize(rows))
    report=paired_comparison(completed['base'],completed['teacher']);report['elapsed_s']=time.monotonic()-started
    write(out/'comparison.json',report);write(out/'complete.json',dict(time=time.time(),**report))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()





