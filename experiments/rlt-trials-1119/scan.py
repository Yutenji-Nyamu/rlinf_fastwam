import copy,json,sys,time,os
from pathlib import Path
def main():
    d=Path(sys.argv[1]); h=int(sys.argv[2]);os.environ['CUDA_VISIBLE_DEVICES']='4'
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs import get_env_cls
    from rlinf.models.embodiment.openpi import get_model
    c=OmegaConf.load(d/'config/formal.yaml')
    f=c.rollout.rlt_feature_model
    f.num_action_chunks=50;f.openpi.action_chunk=50
    model=get_model(f).eval().cuda()
    ec=copy.deepcopy(c.env.eval);n=int(ec.total_num_envs)
    assert ec.max_episode_steps==200 and n==48 and h in (10,50)
    ec.auto_reset=False;ec.ignore_terminations=False
    env=get_env_cls(ec.env_type,ec)(ec,n,0,1,None)
    obs,_=env.reset();ids=env.reset_state_ids.tolist()
    success=torch.zeros(n,dtype=torch.bool);done=success.clone()
    records=[];torch.manual_seed(8811);torch.cuda.manual_seed_all(8811)
    try:
     for s in range(0,200,h):
      obs.setdefault("extra_view_images",None)
      with torch.inference_mode(): actions,_=model.predict_action_batch(obs,mode='eval')
      assert actions.shape[:2]==(n,50),actions.shape
      nxt,_,term,trunc,_=env.chunk_step(actions[:,:h]);obs=nxt[-1]
      term=torch.as_tensor(term).bool().reshape(n,-1).any(-1).cpu()
      trunc=torch.as_tensor(trunc).bool().reshape(n,-1).any(-1).cpu()
      success |=term & ~done;done|=term|trunc
      records.append([s+h,int(success.sum())])
      if done.all():break
     result=dict(horizon=h,seed=8811,layouts=ids,success=success.tolist(),history=records)
     (d/f'scan-h{h}.json').write_text(json.dumps(result))
     print(h,records,flush=True)
    finally:env.close()
    os._exit(0)

if __name__=='__main__':main()
