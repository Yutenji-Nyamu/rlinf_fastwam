"""Compare the frozen native VLA to a small feasibility threshold before token training."""
import json,os,subprocess,sys
from pathlib import Path

def evaluate(directory,task):
    plan=json.loads((directory/'plan.json').read_text())
    os.environ['CUDA_VISIBLE_DEVICES']=str(plan['gpu'])
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs import get_env_cls
    from rlinf.models.embodiment.openpi import get_model
    cfg=OmegaConf.load(directory/'choices'/task/'formal.yaml')
    fc=cfg.rollout.rlt_feature_model
    fc.openpi.use_rlt=False
    fc.openpi.rlt_token_path=None
    model=get_model(fc).eval().cuda()
    ec=cfg.env.eval;ec.auto_reset=False;ec.ignore_terminations=False
    n=ec.total_num_envs;h=cfg.actor.model.num_action_chunks
    assert n==48 and ec.max_episode_steps==200 and h==10
    env=get_env_cls(ec.env_type,ec)(ec,n,0,1,None)
    try:
        obs,_=env.reset()
        done=torch.zeros(n,dtype=torch.bool);success=done.clone()
        torch.manual_seed(8811);torch.cuda.manual_seed_all(8811)
        for _ in range(200//h):
            obs.setdefault('extra_view_images',None)
            with torch.inference_mode():actions,_=model.predict_action_batch(obs,mode='eval')
            assert actions.shape[:2]==(n,h)
            nxt,_,term,trunc,_=env.chunk_step(actions)
            term=torch.as_tensor(term).bool().reshape(n,-1).any(-1).cpu()
            trunc=torch.as_tensor(trunc).bool().reshape(n,-1).any(-1).cpu()
            success|=term & ~done;done|=term|trunc;obs=nxt[-1]
            if done.all():break
        result=dict(task=task,episodes=n,successes=int(success.sum()),layouts=env.reset_state_ids.tolist(),success=success.tolist())
        (directory/(task+'-screen.json')).write_text(json.dumps(result));print(task,result['successes'],flush=True)
    finally:env.close()
    os._exit(0)

def choose(directory):
    import shutil,fcntl
    from omegaconf import OmegaConf
    read=lambda f:json.loads(f.read_text())
    plan=read(directory/'plan.json');candidates=plan['screen_tasks'];tested={}
    for task in candidates:
        subprocess.run([sys.executable,'-uB',__file__,str(directory),task],check=True,timeout=3500)
        rec=read(directory/(task+'-screen.json'));tested[task]=rec['successes']
        if rec['successes']>=4:break
    task=max(tested,key=tested.get);c=directory/'choices'/task
    for name in ('formal.yaml','stage1.yaml'):shutil.copy2(c/name,directory/'config'/name)
    plan.update(task=task,stage1_data=read(c/'data.json')['dataset'])
    from common import save
    save(directory/'plan.json',plan);save(directory/'screen-result.json',dict(selected=task,successes=tested,episodes=48))
    active=Path(read(directory/'wait.json')['active_index'])
    with active.with_suffix('.continuation.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);index=read(active);slot=index['slots'][str(plan['gpu'])]
        assert slot['owner']==str(directory);slot['task']=task;save(active,index)
if __name__=='__main__':
    directory=Path(sys.argv[1])
    if len(sys.argv)==3:evaluate(directory,sys.argv[2])
    else:
        sys.path.insert(0,json.loads((directory/'wait.json').read_text())['queue_ops'])
        choose(directory)

