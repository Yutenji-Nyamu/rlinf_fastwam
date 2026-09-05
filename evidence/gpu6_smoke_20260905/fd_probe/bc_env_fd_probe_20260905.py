"""Reproduce the 32+32 scene boundary without a policy or any action step."""
import json
import os
import resource
from pathlib import Path
import torch
from omegaconf import OmegaConf
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

run=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/env-fd-probe-20260905')
run.mkdir(exist_ok=False)
source=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v5/runtime/resolved.yaml')
cfg=OmegaConf.load(source)
torch.set_num_threads(4)
assert torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']=='6'
initial_limit=resource.getrlimit(resource.RLIMIT_NOFILE)
assert initial_limit[0]==1024,initial_limit
events=[]
def record(stage,**extra):
    try: fds=len(os.listdir('/proc/self/fd'))
    except OSError as exc: fds=f'errno={exc.errno}: {exc.strerror}'
    row={'stage':stage,'open_fds':fds,'nofile':resource.getrlimit(resource.RLIMIT_NOFILE),**extra}
    events.append(row)
    print(json.dumps(row),flush=True)
record('begin',physical_gpu=6,policy_queries=0,optimizer_updates=0,max_scenes=64)
envs=[]
passed=False
try:
    for name in ('train','eval'):
        env_cfg=cfg.env[name]
        env_cfg.task_config.save_path=str(run/'robotwin_data')
        env_cfg.video_cfg.save_video=False
        env=RoboTwinEnv(env_cfg,32,0,1,None)
        envs.append(env)
        record(name+'_created')
        try:
            obs,_=env.reset()
            record(name+'_reset_render_passed',image_shape=list(obs['main_images'].shape))
            del obs
        except RuntimeError as exc:
            record(name+'_reset_render_failed',error=str(exc))
            # Only attempt the identified fd boundary, with retained same scenes.
            if 'getSemaphoreFdKHR' not in str(exc): raise
            resource.setrlimit(resource.RLIMIT_NOFILE,(min(4096,initial_limit[1]),initial_limit[1]))
            record('raised_only_this_process_soft_limit')
            raw=env.venv.get_obs()
            record('same_scenes_get_obs_retry_passed',observations=len(raw))
            del raw
    passed=True
finally:
    record('result',passed=passed)
    (run/'result.json').write_text(json.dumps(events,indent=2))
    # Close only the scenes belonging to this isolated diagnostic.
    for env in envs:
        env.offload(clear_cache=False)
