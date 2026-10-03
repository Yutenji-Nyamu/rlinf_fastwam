"""Bounded actual EXPO environment reset; no policy or learner is constructed."""
import json,os,time
from pathlib import Path

def main():
    assert Path('/proc/self/comm').read_text().strip()=='expo-gpu4567'
    from omegaconf import OmegaConf
    from rlinf.algorithms.expo_ft.backend import create_robotwin_env
    import numpy as np
    import torch
    R=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001')
    inputs=json.loads((R/'formal-turn-switch-repair-20261002/inputs.json').read_text())
    cfg=OmegaConf.create(inputs['env'])
    cfg.video_cfg.save_video=False
    cfg.video_cfg.video_base_dir=str(R/'gpu4567-fix-20261003/probe-video')
    cfg.task_config.save_path=str(R/'gpu4567-fix-20261003/probe-env-data')
    env=None
    print(json.dumps({'stage':'before_environment','pid':os.getpid(),'time':time.time()}),flush=True)
    try:
        env=create_robotwin_env(cfg,num_envs=1,seed_offset=0)
        result=env.reset()
        summary=[]
        def walk(x,path):
            if isinstance(x,dict):
                for k,v in x.items():walk(v,path+'/'+str(k))
            elif isinstance(x,(list,tuple)):
                for i,v in enumerate(x):walk(v,path+'/'+str(i))
            elif isinstance(x,(torch.Tensor,np.ndarray)):
                a=x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else x
                summary.append({'path':path,'shape':list(a.shape),'dtype':str(a.dtype),'finite':bool(np.isfinite(a).all()),'std':float(a.std()) if a.size else None})
        walk(result,'reset')
        images=[x for x in summary if len(x['shape'])>=3 and x['shape'][-1]==3]
        assert sum(int(np.prod(x['shape'][:-3])) for x in images)==3 and all(x['finite'] and x['std']>1 for x in images),summary
        print(json.dumps({'stage':'native_reset_verified','pid':os.getpid(),'time':time.time(),'summary':summary,'binding':env.expo_renderer_binding}),flush=True)
        time.sleep(3)
    finally:
        if env is not None:
            from rlinf.algorithms.expo_ft.lifecycle import close_robotwin_env
            close_robotwin_env(env)
    print(json.dumps({'stage':'closed','pid':os.getpid(),'time':time.time()}),flush=True)

if __name__=='__main__':main()
