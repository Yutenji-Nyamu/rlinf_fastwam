"""Fixed-workload online success FM-BC timing.

Uses RLinf OpenPI sampling, masked_fm_loss and the RoboTwin environment adapter.
Each K starts at the same student weights; side samples preserve main-policy RNG.
"""
from helpers import generate, sync_time, write_csv
import argparse,copy,json,math,os,random,resource,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2))


def entropy_weights(samples):
    h=.5*torch.log(2*math.pi*math.e*(samples.float().var(1,unbiased=True)+1e-6)).sum(-1)
    z=((h-h.mean())/h.std(unbiased=False).clamp_min(1e-6)).clamp(-2,2)
    # Existing bounded-linear mapper: z=-2/0/+2 maps to 0/1/5.
    return h,1+.5*z.clamp_max(0)+2*z.clamp_min(0)


def main(job):
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.base_policy import ForwardType
    from tools.teacher_eval.evaluate import environment
    from tools.appendix_signals.dojo_sampling import add_legacy_trace
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,max(8192,soft)),hard))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg=OmegaConf.load(job['config']);mc=copy.deepcopy(cfg.actor.model)
    mc.openpi.use_dsrl=False;mc.openpi.use_rlt=False
    teacher=get_model(mc).cuda().eval().requires_grad_(False)
    traces=[]
    sampler=teacher.sample_actions
    def traced(observation, *args, **kwargs):
        noise=kwargs.get('noise')
        traces.append([len(observation.state), list(noise.shape) if noise is not None else None])
        return sampler(observation,*args,**kwargs)
    teacher.sample_actions=traced
    add_legacy_trace(teacher)
    student=get_model(mc).cuda();params=[p for p in student.parameters()if p.requires_grad]
    for p in params:p.data=p.data.float()
    initial=[p.detach().cpu().clone()for p in params]
    # Verify the real weighted FM update before paying for simulator initialization.
    with np.load(job['preflight_inputs'],allow_pickle=False)as z:
        ids=np.arange(job['micro_batch'])%len(z['states'])
        ob={key:torch.from_numpy(z[key][ids].copy())for key in ['main_images','wrist_images','states']}
        ob.update(task_descriptions=z['task_descriptions'][ids].tolist(),extra_view_images=None)
    with torch.no_grad():act,ret=teacher.predict_action_batch(ob,mode='eval',compute_values=False)
    batch={key:value.detach().cpu().clone()for key,value in ret['forward_inputs'].items()if torch.is_tensor(value)}
    batch.update(action=act.cpu(),model_action=ret['model_actions'].cpu())
    prepared=student.prepare_dagger_sft_batch(batch);count=act.shape[1]
    probe_opt=torch.optim.AdamW(params,lr=2.5e-5,fused=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        probe_loss=student(forward_type=ForwardType.SFT,data=prepared,use_action_chunk_loss=True,
            action_valid_mask=torch.ones((len(ids),count,14),device='cuda',dtype=torch.bool),action_weights=torch.ones((len(ids),count),device='cuda'))
    assert bool(torch.isfinite(probe_loss));probe_loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
    assert torch.isfinite(norm)and norm>0;probe_opt.step();probe_opt.zero_grad(set_to_none=True)
    write(out/'update-preflight.json',dict(loss=float(probe_loss.detach()),gradient_norm=float(norm),optimizer_step=True))
    with torch.no_grad():
        for p,v in zip(params,initial):p.copy_(v)
    del probe_opt,probe_loss,prepared,batch,act,ret,ob;torch.cuda.empty_cache()
    n=job['parallel'];chunk=int(mc.num_action_chunks);steps=int(cfg.env.eval.max_episode_steps)
    manifest=dict(algorithm='Online success FM-BC, fixed update workload',task=cfg.env.eval.task_config.task_name,
        gpu=torch.cuda.get_device_name(),parallel=n,trajectories_per_round=32,denoise_steps=10,chunk=chunk,
        action_budget=steps,optimizer_steps=job['updates'],global_batch=job['batch'],micro_batch=job['micro_batch'],
        warmup_rounds=1,measured_rounds=3,K=[1,2,4,8],scheduling='single N*K call per query, env-major, execute candidate zero; paired replay outside round',
        weighting='K1 unit; K>1 diagonal Gaussian entropy, query z-score across all 32 trajectories, bounded linear 0/1/5',
        update_scope='successful online action buffer; same initialization and sampling schedule across K; updated actor synchronized each round',trainable_parameters=sum(p.numel()for p in params))
    write(out/'protocol.json',manifest);rows=[];paired=[];flat=[];calls=[]
    for k in [1,2,4,8]:
        with torch.no_grad():
            for p,v in zip(params,initial):p.copy_(v)
            for (name,p) in student.named_parameters():
                if p.requires_grad:dict(teacher.named_parameters())[name].copy_(p)
        opt=torch.optim.AdamW(params,lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,fused=True)
        for repeat in range(4):
            timings={key:0. for key in ['environment','policy','entropy_weights','update']}
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            seed=17111700+repeat;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            write(out/'progress.json',dict(K=k,repeat=repeat,phase='environment_reset',time=time.time()))
            records=[]; valid=[]; successes=0; replay=[]; raw_h=[]
            copy_seconds=0.
            for wave,lo in enumerate(range(0,32,n)):
                t=sync_time()
                env=environment(cfg,job['seeds'][lo:lo+n])
                success=np.zeros(n,bool)
                try:
                    obs,_=env.reset();timings['environment']+=sync_time()-t
                    for q in range(steps//chunk):
                        obs.setdefault('extra_view_images',None)
                        noise_seed=(seed*64+wave*8+q)*16
                        if repeat and k>1:
                            t=sync_time()
                            replay.append((wave,q,noise_seed,{a:(v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for a,v in obs.items()}))
                            copy_seconds+=sync_time()-t
                        traces.clear();t=sync_time()
                        with torch.no_grad():actions,main,samples,forward=generate(teacher,obs,k,noise_seed)
                        timings['policy']+=sync_time()-t
                        assert traces==[[n*k,[n*k,teacher.config.action_horizon,teacher.config.action_dim]]],traces
                        if repeat==0 and wave==q==0:
                            calls.append(dict(K=k,envs=n,calls=len(traces),input_shape=traces[0][1],output_shape=list(samples.shape),executed=0))
                            write(out/'calls.json',calls)
                        t=sync_time()
                        if k>1:
                            h=.5*torch.log(2*math.pi*math.e*(samples.float().var(1,unbiased=True)+1e-6)).sum(-1)
                        else:h=torch.zeros((n,chunk),device='cuda')
                        raw_h.append(h);timings['entropy_weights']+=sync_time()-t
                        record={a:b.detach().cpu().clone() for a,b in forward.items()}
                        record.update(model_action=main.cpu().clone(),action=actions.cpu().clone())
                        del forward,main,samples
                        t=sync_time()
                        before=env.elapsed_steps.detach().cpu().clone()
                        nxt,_,_,_,_=env.chunk_step(actions);after=env.elapsed_steps.detach().cpu().clone()
                        timings['environment']+=sync_time()-t
                        record['action_valid_mask']=(torch.arange(chunk)[None,:]<(after-before)[:,None])[:,:,None].expand(n,chunk,14)
                        records.append(record);obs=nxt[-1];success|=env.success_once.cpu().numpy().astype(bool)
                        write(out/'progress.json',dict(K=k,repeat=repeat,wave=wave,query=q,time=time.time()))
                finally:
                    t=sync_time();env.close();timings['environment']+=sync_time()-t
                successes+=int(success.sum())
                for idx in range(wave*(steps//chunk),(wave+1)*(steps//chunk)):
                    valid.extend((idx,i) for i in range(n) if success[i] and records[idx]['action_valid_mask'][i].any())
            t=sync_time()
            for q in range(steps//chunk):
                idx=list(range(q,len(records),steps//chunk))
                h=torch.cat([raw_h[i] for i in idx])
                z=((h-h.mean())/h.std(unbiased=False).clamp_min(1e-6)).clamp(-2,2)
                weights=1+.5*z.clamp_max(0)+2*z.clamp_min(0) if k>1 else torch.ones_like(h)
                for i,w in zip(idx,weights.split(n)):records[i]['action_weights']=w[:,:chunk].cpu()
            timings['entropy_weights']+=sync_time()-t
            # Fixed number of optimizer updates: avoids success-rate-dependent update budgets.
            t=time.perf_counter();student.train();rng=np.random.default_rng(seed)
            if not valid:raise RuntimeError('No successful episodes for FM-BC; incomplete timing round')
            losses=[]
            for update in range(job['updates']):
                opt.zero_grad(set_to_none=True);ids=rng.integers(len(valid),size=job['batch'])
                for lo in range(0,len(ids),job['micro_batch']):
                    selected=[valid[x]for x in ids[lo:lo+job['micro_batch']]]
                    batch={key:torch.stack([records[q][key][i]for q,i in selected])for key in records[0]if torch.is_tensor(records[0][key])}
                    prepared=student.prepare_dagger_sft_batch(batch)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=student(forward_type=ForwardType.SFT,data=prepared,use_action_chunk_loss=True,action_valid_mask=batch['action_valid_mask'].cuda(),action_weights=batch['action_weights'].cuda())
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite FM-BC loss')
                    (loss*len(selected)/job['batch']).backward();losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            torch.cuda.synchronize();timings['update']=time.perf_counter()-t
            # Teacher still holds the rollout weights. Replay before copying the update.
            rollout_peak=torch.cuda.max_memory_allocated()/2**30
            before_replay=sync_time()
            measured=dict()
            if replay:
                with torch.random.fork_rng(devices=[0]),torch.no_grad():
                    for wave,q,noise_seed,saved in replay:
                        order=[1,k] if (repeat+wave+q)%2 else [k,1]
                        elapsed={}
                        for variant in order:
                            t=sync_time();result=generate(teacher,saved,variant,noise_seed);elapsed[variant]=sync_time()-t
                            del result
                        paired.append(dict(K=k,repeat=repeat,wave=wave,query=q,first=order[0],k1_s=elapsed[1],k_s=elapsed[k],delta_s=elapsed[k]-elapsed[1]))
                        measured[wave,q]=elapsed[k]-elapsed[1]
            after_replay=sync_time()
            with torch.no_grad():
                student_params=dict(student.named_parameters())
                for name,p in teacher.named_parameters():
                    source=student_params[name]
                    if source.requires_grad:p.copy_(source)
            end=sync_time()
            total=end-start-(after_replay-before_replay)-copy_seconds
            row=dict(K=k,repeat=repeat,warmup=repeat==0,wall_seconds=total,other_seconds=total-sum(timings.values()),
                peak_allocated_gib=rollout_peak,seconds=timings,successes=successes,updates=job['updates'])
            flat.append(dict(K=k,repeat=repeat,warmup=repeat==0,envs=n,episodes=32,wall_s=total,**timings,
                other_s=row['other_seconds'],paired_delta_s=sum(measured.values()),paired_share=sum(measured.values())/total,
                peak_allocated_gib=rollout_peak,successes=successes,updates=job['updates'],start=start,pre_replay=before_replay,post_replay=after_replay,end=end))
            write_csv(out/'rounds.csv',flat)
            if paired:write_csv(out/'paired.csv',paired)
            rows.append(row);write(out/'rounds.json',rows);print(json.dumps(row),flush=True)
            del records,batch,prepared,loss;opt.zero_grad(set_to_none=True)
        del opt;torch.cuda.empty_cache()
    from report import render
    render(out)
    write(out/'complete.json',dict(rounds=16,measured=12,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();main(json.loads(Path(a.job).read_text()))
