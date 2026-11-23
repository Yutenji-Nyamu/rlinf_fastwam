"""Run from the same GPU-scoped environment as the trainer."""
import json, os, sys, time
from pathlib import Path
b=Path(os.environ["RLINF_REPAIR_DEPLOYMENT"])
read=lambda p:json.loads(p.read_text())

def exercise(g,case,width):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.robotwin.sharded_vector_env import ShardedVectorEnv
    d=b if case=='fast' else b/case
    cfg=OmegaConf.load(d/'config/formal.yaml').env.train
    assert torch.cuda.device_count()==1
    settings=OmegaConf.to_container(cfg.task_config,resolve=True)
    settings['save_path']=str(b/'probe-output'/f'{g}-{case}-{width}')
    seeds=np.random.default_rng(0).integers(0,200000,width).tolist()
    seeds[:4]=[188379,153778,68239,184933][:width]
    if case=='fast':
        ref=read(Path(cfg.seeds_path))
        if isinstance(ref,dict):
            ref=ref[cfg.task_config.task_name]
            if isinstance(ref,dict):ref=ref['success_seeds']
        seeds=[int(ref[0])]*width
    if width>4:
        from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
        cfg.task_config.save_path=settings['save_path']
        cfg.video_cfg.save_video=False
        start=time.monotonic()
        adapter=RoboTwinEnv(cfg,width,int(case=='fast' and g==7),2 if case=='fast' else 1,None)
        assert isinstance(adapter.venv,ShardedVectorEnv)
        try:
            obs,_=adapter.reset()
            assert obs['states'].shape==(width,14) and len(obs['main_images'])==width
            assert obs['wrist_images'].shape[:2]==(width,2)
            instructions=obs['task_descriptions']
            assert all(instructions)
            if case=='fast':
                for i in range(0,width,8):
                    assert len(set(instructions[i:i+8]))==1
            horizon=24 if case=='fast' else 10
            result=adapter.chunk_step(obs['states'][:,None,:].repeat(1,horizon,1))
            assert len(result)==5 and result[1].shape==(width,horizon)
            assert torch.isfinite(result[1]).all()
        finally:
            adapter.offload()
        record=dict(task=settings['task_name'],gpu=g,width=width,shards=int(cfg.native_shards),
                    entry='RoboTwinEnv.reset/chunk_step/offload',action_steps=horizon,
                    total_seconds=round(time.monotonic()-start,2))
        (b/f'probe-{case}-{g}-{width}.json').write_text(json.dumps(record))
        print(json.dumps(record),flush=True)
        return
    env=ShardedVectorEnv(n_envs=width,env_seeds=seeds,shards=cfg.native_shards,task_config=settings)
    times={}; snapshots=[]
    start=time.monotonic()
    def reset_and_step():
        t=time.monotonic()
        env.reset(env_seeds=seeds)
        obs=env.get_obs()
        assert len(obs)==width
        for o in obs:
            assert o['instruction'] and np.asarray(o['state']).shape==(14,)
            for key in ('full_image','left_wrist_image','right_wrist_image'):
                assert o[key].ndim==3 and o[key].shape[-1]==3
        snapshots.append([o['instruction'] for o in obs])
        if case=='fast':assert len(set(snapshots[-1]))==1
        action=np.stack([o['state'] for o in obs])[:,None,:]
        result=env.step(action)
        assert len(result)==5 and all(len(r)==width for r in result)
        assert np.isfinite(np.asarray(result[1])).all()
        return round(time.monotonic()-t,2)
    try:
        times['first']=reset_and_step()
        if width==4:
            env.reset(env_idx=[3,0],env_seeds=[seeds[3],seeds[0]])
            assert len(env.get_obs())==width
            env.close()
            times['after_offload']=reset_and_step()
            assert snapshots[0]==snapshots[1]
    finally:
        env.close()
    record=dict(task=settings['task_name'],gpu=g,width=width,shards=int(cfg.native_shards),
                timings=times,total_seconds=round(time.monotonic()-start,2))
    (b/f'probe-{case}-{g}-{width}.json').write_text(json.dumps(record))
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    exercise(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
