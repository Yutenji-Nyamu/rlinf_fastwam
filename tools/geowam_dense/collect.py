"""Dense GeoWAM offline replay; preserve independent K4 samples from the G01 collector."""
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from plot_episode import render


def write_json(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    tmp.replace(path)


def run(job):
    import torch
    torch.set_num_threads(1)
    os.environ['GEOWAM_WORKSPACE']=job['workspace']
    sys.path[:0]=[job['repo']+'/src',job['repo']+'/scripts/geowam']
    from pipeline_common import episode_meta,read_rgb,read_depth
    from ur5e_real.adapters.geowam.masks import Sam31Masker
    from ur5e_real.adapters.geowam.policy import GeoWAMPolicy
    spec=importlib.util.spec_from_file_location('g01_sampler',job['collector'])
    sampler=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    masker=Sam31Masker(job['sam'],Path(job['bundle'])/'perception.json',device='cuda:0')
    policy=GeoWAMPolicy(Path(job['bundle']),device='cuda:0',vae=job['vae'],masker=masker)
    root=Path(job['output'])
    slot=job.get('worker',os.environ['CUDA_VISIBLE_DEVICES'])
    failures=[]
    for row in job['episodes']:
        folder=root/row['id'];folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():
            continue
        try:
            meta=episode_meta(row)
            raw=Path(job['workspace'])/'data/raw'/row['path']
            with (raw/'sync.csv').open() as f:
                sync=list(csv.DictReader(f))
            names=meta['names'];n=len(names)
            assert len(sync)==n==row['frames'] and all(s['head_image']==name for s,name in zip(sync,names))
            cache=folder/'working';cache.mkdir(exist_ok=True)
            normal=np.full((n,4,32,7),np.nan,np.float32);physical=np.full_like(normal,np.nan)
            first_entropy=np.full(n,np.nan);valid=np.zeros(n,bool)
            for i,name in enumerate(names):
                frame=cache/f'{i:04d}.npz'
                if frame.exists():
                    with np.load(frame) as stored:
                        if 'model' in stored:
                            normal[i],physical[i]=stored['model'],stored['physical']
                            valid[i]=True
                else:
                    obs=dict(head=read_rgb(row,name,'head'),wrist=read_rgb(row,name,'wrist'),
                             depth_m=read_depth(row,name,meta['depth_scale']),
                             state=meta['state'][i].copy(),task=row['task'])
                    try:
                        encoded=policy.encode(obs)
                    except ValueError as exc:
                        if 'depth' not in str(exc).lower():raise
                        data={}
                    else:
                        seed=171117000+int(row['id'][1:])*1000+i
                        a,b=sampler.sample_many(policy,encoded,obs['state'],4,seed)
                        assert a.shape==b.shape==(4,32,7) and np.isfinite(a).all() and np.isfinite(b).all()
                        normal[i],physical[i],valid[i]=a,b,True
                        data={'model':a,'physical':b}
                    tmp=frame.with_suffix('.tmp.npz');np.savez(tmp,**data);tmp.replace(frame)
                if valid[i]:
                    variance=normal[i,:,0,:].astype(np.float64).var(axis=0,ddof=1)
                    first_entropy[i]=.5*np.log(2*np.pi*np.e*(variance+1e-6)).sum()
                if i%10==0 or i==n-1:
                    write_json(root/f'progress-g{slot}.json',dict(episode=row['id'],frame=i+1,frames=n,time=time.time()))
            np.savez_compressed(folder/'samples.npz',model_actions=normal,env_actions=physical,state=meta['state'],
                                first_step_entropy=first_entropy,depth_valid=valid,time_s=meta['time']-meta['time'][0],
                                video_frame=np.array([int(s['frame_idx']) for s in sync]),action_step=np.arange(1,n+1))
            recap=render(folder,row,sync,[raw/'images/head'/name for name in names],meta['time'],first_entropy,valid)
            assert valid.any()
            write_json(folder/'complete.json',recap)
            print('complete',row['id'],row['key'],int(valid.sum()),n,flush=True)
        except Exception as exc:
            import traceback
            failures.append(dict(id=row['id'],key=row['key'],error=repr(exc)))
            write_json(root/f'failures-g{slot}.json',failures)
            traceback.print_exc()
    if failures:raise RuntimeError(f'{len(failures)} episode(s) require retry')
    write_json(root/f'complete-g{slot}.json',dict(episodes=len(job['episodes']),time=time.time()))


if __name__=='__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
