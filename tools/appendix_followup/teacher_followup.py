"""Frozen teacher shadow queries and paired first-three-chunk entropy interventions.

Uses teacher_eval checkpoint loaders/environment; first3 protocol follows
pi05_policy_eval.protocol.choose and threshold_sweep (strict quantile gates).
Action RMSE follows pillbottle_slot25_shadow/teacher_shadow.py.
"""
import sys,json,dataclasses,time,random,resource
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)

def gate(score,direction,threshold,query,active):
    if query>=3:return np.zeros_like(active)
    selected=score>threshold if direction=='high' else score<threshold
    return selected&active

def inference(models,kind,condition,obs):
    if kind=='dsrl':
        m=models['model'];m.config=dataclasses.replace(m.config,use_dsrl=condition=='teacher')
        a,r=m.predict_action_batch(obs,mode='eval',compute_values=False)
        return a,r['model_actions'][...,:14]
    f=models['feature'];features,context=f.extract_rlt_obs(obs,return_decode_context=True)
    if condition=='base':raw=features['ref_chunk']
    else:raw,_=models['model'].predict_action_batch(env_obs=features,mode='eval',return_obs=True)
    return f.decode_rlt_action(raw,context),raw

def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def render(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=[]
    for p in (out/'calibration').glob('q*.npz'):
        with np.load(p)as z:
            valid=z['valid'];values.extend(zip(z['rmse'][valid].tolist(),z['entropy'][valid].tolist()))
    if values:
        a=np.array(values);idx=np.argsort(a[:,0]);bins=[a[x].mean(0)for x in np.array_split(idx,10)if len(x)];b=np.array(bins)
        fig,ax=plt.subplots(figsize=(5,3.6),layout='constrained');ax.scatter(a[:,0],a[:,1],s=3,alpha=.15);ax.plot(b[:,0],b[:,1],'o-',color='#d15a32');ax.set(xlabel='Teacher–base action RMSE (normalized D14)',ylabel='Base action entropy')
        for ext in ['png','pdf','svg']:fig.savefig(out/f'action-difference.{ext}',dpi=180)
        np.savez_compressed(out/'scatter.npz',rmse=a[:,0],entropy=a[:,1],equal_count_bins=b);plt.close(fig)
    reports=[json.loads(p.read_text())for p in (out/'conditions').glob('*.json')]
    if reports:
        fig,ax=plt.subplots(figsize=(5,3.6),layout='constrained')
        base=next((r for r in reports if r['direction']=='base'),None)
        full=next((r for r in reports if r['direction']=='full'),None)
        for direction,color in [('high','#bd4839'),('low','#2986a4')]:
            rr=[r for r in reports if r['direction']==direction]+([base]if base else [])+([full]if full else []);rr.sort(key=lambda r:r['actual_rate'])
            if rr:ax.plot([r['actual_rate']*100 for r in rr],[r['success_rate']*100 for r in rr],'o-',label=direction,color=color)
        ax.set(xlabel='Actual takeover among eligible first-3 queries (%)',ylabel='Success rate (%)',ylim=(0,100));ax.legend()
        for ext in ['png','pdf','svg']:fig.savefig(out/f'takeover.{ext}',dpi=180)
        plt.close(fig);write(out/'summary.json',reports)

def run(j):
    from tools.teacher_eval.evaluate import build_models,environment,predict
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,max(soft,8192)),hard))
    cfg=OmegaConf.load(j['config']);models=build_models(cfg,j['kind'],j['checkpoint']);C=j['chunk_steps'];n=len(j['seeds']);out=Path(j['output']);out.mkdir(parents=True,exist_ok=True)
    for k in ['model','feature']:
        if k in models:models[k].cuda()
    write(out/'protocol.json',dict(kind=j['kind'],chunk=C,budget=200,parallel=n,episodes_per_condition=n,max_takeover_queries=3,targets=list(range(10,100,10))+[100],entropy='diagonal Gaussian K4, unbiased variance, eps1e-6, normalized D14',score='mean entropy over execution prefix',distance='per-action RMSE over normalized D14; raw normalized and physical actions saved',calibration_noise_seed=j['noise_seed']+1000000,evaluation_noise_seed=j['noise_seed'],source='teacher_eval; pi05_policy_eval.protocol/threshold_sweep; teacher_shadow.py'))
    def rollout(name,direction,target,threshold=None):
        done=out/'conditions'/f'{name}.json'
        if done.exists():return json.loads(done.read_text())
        write(out/'progress.json',dict(condition=name,phase='reset',time=time.time()))
        env=environment(cfg,j['seeds']);success=np.zeros(n,bool);eligible=np.zeros(n,int);taken=np.zeros(n,int);scores=[];queries=[];writer=None
        try:
            obs,_=env.reset();initial=env.venv._call('episode_state')
            for q in range(200//C):
                obs.setdefault('extra_view_images',None);rng=j['noise_seed']+(1000000 if direction=='calibration'else 0)+q*101;seed(rng)
                with torch.inference_mode():base,braw=inference(models,j['kind'],'base',obs)
                valid_before=~success & (env.elapsed_steps.cpu().numpy()<200);h=None;teacher=None
                if direction=='calibration' or (q<3 and direction not in ['base','full']):
                    with torch.random.fork_rng(devices=[0]),torch.inference_mode():
                        seed(rng+5000000);samples=[braw[:,:C].float()]
                        for k in range(3):_,raw=inference(models,j['kind'],'base',obs);samples.append(raw[:,:C].float())
                        stack=torch.stack(samples,1);h=(.5*torch.log(2*np.pi*np.e*(stack.var(1,unbiased=True)+1e-6))).sum(-1).cpu().numpy()
                        score=h.mean(-1)
                        if direction=='calibration':seed(rng);teacher,traw=inference(models,j['kind'],'teacher',obs)
                    if direction=='calibration' and q==0:
                        with torch.random.fork_rng(devices=[0]),torch.inference_mode():
                            seed(rng);original,_=predict(models,j['kind'],'teacher',obs)
                            torch.testing.assert_close(teacher,original,rtol=0,atol=1e-5)
                        write(out/'teacher-parity.json',dict(max_abs=float((teacher-original).abs().max()),passed=True))
                else:score=np.zeros(n)
                if direction=='calibration' and q<3:scores.extend(score[valid_before].tolist())
                select=(valid_before.copy() if direction=='full' and q<3 else gate(score,direction,threshold,q,valid_before) if direction in ['high','low'] else np.zeros(n,bool))
                if select.any():
                    with torch.random.fork_rng(devices=[0]),torch.inference_mode():seed(rng);teacher,traw=inference(models,j['kind'],'teacher',obs)
                action=base.clone()
                if select.any():action[torch.as_tensor(select,device=action.device)]=teacher.to(action.device)[torch.as_tensor(select,device=action.device)]
                if q<3:eligible+=valid_before;taken+=select
                # Save query-cadence representative video, with explicit action indices.
                if name in ['base','high-50','low-50','full']:
                    import cv2
                    im=obs['main_images'][0].cpu().numpy()
                    if im.shape[0]==3:im=np.moveaxis(im,0,-1)
                    if writer is None:writer=cv2.VideoWriter(str(out/f'{name}-query-frames.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2,(im.shape[1],im.shape[0]))
                    writer.write(cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_RGB2BGR))
                before=env.elapsed_steps.cpu().numpy().copy();result=env.chunk_step(action);after=env.elapsed_steps.cpu().numpy().copy();valid=np.arange(C)[None,:]<(after-before)[:,None]
                if direction=='calibration':
                    d=out/'calibration';d.mkdir(exist_ok=True);delta=(traw[:,:C].float()-braw[:,:C].float()).cpu().numpy()
                    np.savez_compressed(d/f'q{q:03d}.npz',base_model=braw[:,:C].float().cpu().numpy(),teacher_model=traw[:,:C].float().cpu().numpy(),base_env=base.cpu().numpy(),teacher_env=teacher.cpu().numpy(),entropy=h,rmse=np.sqrt((delta**2).mean(-1)),samples=stack.cpu().numpy(),valid=valid,before=before,after=after)
                    np.savez_compressed(d/f'obs{q:03d}.npz',**{k:v.cpu().numpy()for k,v in obs.items()if torch.is_tensor(v)},task_descriptions=np.asarray(obs['task_descriptions']))
                meta=env.venv._call('episode_state');success|=np.array([x['success']for x in meta])|env.success_once.cpu().numpy().astype(bool);obs=result[0][-1]
                queries.append(dict(query=q,action_before=before.tolist(),action_after=after.tolist(),entropy=score.tolist(),selected=select.tolist(),eligible=valid_before.tolist() if q<3 else [False]*n))
                write(out/'progress.json',dict(condition=name,phase='rollout',query=q,success=int(success.sum()),time=time.time()))
            episodes=[dict(scene_seed=int(j['seeds'][i]),**initial[i],success=bool(success[i]),eligible=int(eligible[i]),takeovers=int(taken[i]))for i in range(n)]
            report=dict(direction=direction,target=target,threshold=threshold,success_rate=float(success.mean()),actual_rate=float(taken.sum()/max(1,eligible.sum())),all_query_rate=float(taken.sum()/(n*(200//C))),episodes=episodes,queries=queries)
            if direction=='calibration':write(out/'calibration-scores.json',scores)
            write(done,report);return report
        finally:
            if writer is not None:writer.release()
            env.offload(clear_cache=True)
    rollout('calibration','calibration',0);render(out)
    pool=np.array(json.loads((out/'calibration-scores.json').read_text()));assert len(pool)>0 and np.isfinite(pool).all()
    thresholds={str(p):dict(high=float(np.quantile(pool,1-p/100)),low=float(np.quantile(pool,p/100)))for p in range(10,100,10)};write(out/'thresholds.json',thresholds)
    rollout('base','base',0);rollout('full','full',100)
    for p in [50,10,90,20,80,30,70,40,60]:
        for direction in ['high','low']:rollout(f'{direction}-{p}',direction,p,thresholds[str(p)][direction]);render(out)
    write(out/'complete.json',dict(time=time.time(),conditions=20,episodes_per_condition=n))
if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
