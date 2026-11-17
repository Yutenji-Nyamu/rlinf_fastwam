"""Fixed-workload online success FM-BC timing.

Uses RLinf OpenPI sampling, masked_fm_loss and the RoboTwin environment adapter.
Each K starts at the same student weights; side samples preserve main-policy RNG.
"""
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
    teacher=get_model(mc).cuda().eval().requires_grad_(False);add_legacy_trace(teacher)
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
        gpu=torch.cuda.get_device_name(),parallel=n,trajectories_per_round=n,denoise_steps=10,chunk=chunk,
        action_budget=steps,optimizer_steps=job['updates'],global_batch=job['batch'],micro_batch=job['micro_batch'],
        warmup_rounds=1,measured_rounds=3,K=[1,2,4,8],scheduling='synchronous; wall time and module sums both recorded',
        weighting='K1 unit; K>1 diagonal Gaussian entropy, round z-score, bounded linear 0/1/5',
        update_scope='successful online action buffer; same initialization and sampling schedule across K; updated actor synchronized each round',trainable_parameters=sum(p.numel()for p in params))
    write(out/'protocol.json',manifest);rows=[]
    for k in [1,2,4,8]:
        with torch.no_grad():
            for p,v in zip(params,initial):p.copy_(v)
            for (name,p) in student.named_parameters():
                if p.requires_grad:dict(teacher.named_parameters())[name].copy_(p)
        opt=torch.optim.AdamW(params,lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,fused=True)
        for repeat in range(4):
            timings={key:0. for key in ['environment','policy','candidates','entropy_weights','update']}
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            seed=17111700+repeat;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            write(out/'progress.json',dict(K=k,repeat=repeat,phase='environment_reset',time=time.time()))
            t=time.perf_counter();env=environment(cfg,job['seeds'][:n]);obs,_=env.reset();timings['environment']+=time.perf_counter()-t
            records=[];success=np.zeros(n,bool)
            try:
                for q in range(steps//chunk):
                    obs.setdefault('extra_view_images',None)
                    t=time.perf_counter()
                    with torch.no_grad():actions,ret=teacher.predict_action_batch(obs,mode='eval',compute_values=False)
                    torch.cuda.synchronize();timings['policy']+=time.perf_counter()-t
                    main=ret['model_actions'].detach().clone();samples=[main[...,:14]]
                    t=time.perf_counter()
                    if k>1:
                        with torch.random.fork_rng(devices=[0]),torch.no_grad():
                            torch.manual_seed(seed+q*100+k)
                            for _ in range(k-1):
                                _,extra=teacher.predict_action_batch(obs,mode='eval',compute_values=False)
                                samples.append(extra['model_actions'].detach()[...,:14])
                            del extra
                        torch.cuda.synchronize()
                    timings['candidates']+=time.perf_counter()-t if k>1 else 0.
                    t=time.perf_counter()
                    if k>1:h,w=entropy_weights(torch.stack(samples,1))
                    else:w=torch.ones((n,chunk),device='cuda');h=torch.zeros_like(w)
                    torch.cuda.synchronize();timings['entropy_weights']+=time.perf_counter()-t
                    record={a:(b.detach().cpu().clone()if torch.is_tensor(b)else copy.deepcopy(b))for a,b in ret['forward_inputs'].items()}
                    record.update(model_action=main.cpu().clone(),action=actions.cpu().clone(),action_weights=w[:,:chunk].cpu())
                    del samples,ret,main,h,w
                    t=time.perf_counter();before=env.elapsed_steps.detach().cpu().clone();nxt,_,term,trunc,info=env.chunk_step(actions);after=env.elapsed_steps.detach().cpu().clone()
                    timings['environment']+=time.perf_counter()-t
                    mask=(torch.arange(chunk,device='cpu')[None,:]<(after-before)[:,None])[:,:,None].expand(n,chunk,14)
                    record['action_valid_mask']=mask;records.append(record);obs=nxt[-1]
                    success|=env.success_once.cpu().numpy().astype(bool)
                    write(out/'progress.json',dict(K=k,repeat=repeat,query=q,phase='rollout',time=time.time()))
            finally:
                t=time.perf_counter();env.close();timings['environment']+=time.perf_counter()-t
            # Fixed number of optimizer updates: avoids success-rate-dependent update budgets.
            t=time.perf_counter();student.train();rng=np.random.default_rng(seed)
            valid=[(q,i)for q,r in enumerate(records)for i in range(n)if success[i] and r['action_valid_mask'][i].any()]
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
            with torch.no_grad():
                for name,p in student.named_parameters():
                    if p.requires_grad:dict(teacher.named_parameters())[name].copy_(p)
            torch.cuda.synchronize()
            total=time.perf_counter()-start
            row=dict(K=k,repeat=repeat,warmup=repeat==0,started_unix=time.time()-total,ended_unix=time.time(),wall_seconds=total,module_sum_seconds=sum(timings.values()),
                other_seconds=total-sum(timings.values()),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                seconds=timings,successes=int(success.sum()),loss=float(np.mean(losses)),updates=job['updates'])
            rows.append(row);write(out/'rounds.json',rows);print(json.dumps(row),flush=True)
            del records,batch,prepared,loss;opt.zero_grad(set_to_none=True)
        del opt;torch.cuda.empty_cache()
    write(out/'complete.json',dict(rounds=16,measured=12,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();main(json.loads(Path(a.job).read_text()))
