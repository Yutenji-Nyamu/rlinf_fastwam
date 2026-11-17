"""Dense offline inference along recorded RGB-D; one predicted action per observed step."""
import sys,os,json,importlib.util,csv,time
from pathlib import Path
import numpy as np

def run(j):
    import torch,cv2
    torch.set_num_threads(1)
    os.environ['GEOWAM_WORKSPACE']=j['workspace'];sys.path[:0]=[j['repo']+'/src',j['repo']+'/scripts/geowam']
    from pipeline_common import episode_meta,read_rgb,read_depth
    from ur5e_real.adapters.geowam.masks import Sam31Masker
    from ur5e_real.adapters.geowam.policy import GeoWAMPolicy
    spec=importlib.util.spec_from_file_location('existing_collector',j['collector']);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    policy=GeoWAMPolicy(Path(j['bundle']),device='cuda:0',vae=j['vae'],masker=Sam31Masker(j['sam'],Path(j['bundle'])/'perception.json',device='cuda:0'))
    out=Path(j['output']);out.mkdir(parents=True,exist_ok=True)
    for ei,row in enumerate(json.loads(Path(j['manifest']).read_text())[:2]):
        meta=episode_meta(row);dest=out/f'G{ei:02d}';dest.mkdir(exist_ok=True);curve=[];mapping=[];writer=None
        try:
            for s in range(len(meta['names'])):
                head=read_rgb(row,meta['names'][s],'head')
                if writer is None:writer=cv2.VideoWriter(str(dest/'recording.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(head.shape[1],head.shape[0]))
                writer.write(cv2.cvtColor(head,cv2.COLOR_RGB2BGR))
                if s==len(meta['names'])-1:continue
                p=dest/f'q{s:04d}.npz';obs=dict(head=head,wrist=read_rgb(row,meta['names'][s],'wrist'),depth_m=read_depth(row,meta['names'][s],meta['depth_scale']),state=meta['state'][s].copy(),task=row['task'])
                if p.exists():
                    with np.load(p)as z:h=float(z['entropy'][0])
                else:
                    try:encoded=policy.encode(obs)
                    except ValueError as e:
                        if 'depth' not in str(e):raise
                        curve.append(float('nan'));mapping.append(dict(action_step=s+1,query=s,observation_frame=s,target_frame=s+1,reason=str(e),offline_prefix=1,actual_policy_executed_steps=None));continue
                    normal,physical=c.sample_many(policy,encoded,obs['state'],4,171117000+ei*1000+s)
                    hvec=.5*np.log(2*np.pi*np.e*(normal.astype(np.float64).var(0,ddof=1)+1e-6)).sum(-1);h=float(hvec[0])
                    np.savez_compressed(p,model_actions=normal,env_actions=physical,entropy=hvec,head=obs['head'],wrist=obs['wrist'],depth_m=obs['depth_m'],state=obs['state'],recorded_next_state=meta['state'][s+1],offline_prefix=1)
                curve.append(h);mapping.append(dict(action_step=s+1,query=s,observation_frame=s,target_frame=s+1,observation_time_s=float(meta['time'][s]-meta['time'][0]),target_time_s=float(meta['time'][s+1]-meta['time'][0]),offline_prefix=1,actual_policy_executed_steps=None))
                (out/'progress.json').write_text(json.dumps(dict(episode=ei,query=s,total=len(meta['names'])-1,time=time.time())))
        finally:
            if writer is not None:writer.release()
        np.savez_compressed(dest/'curve.npz',action=np.arange(1,len(curve)+1),entropy=curve)
        (dest/'mapping.json').write_text(json.dumps(mapping,indent=2));(dest/'protocol.json').write_text(json.dumps(dict(source='recorded demonstration RGB-D',mode='offline dense re-query, first predicted action at every recorded observation',K=4,H=32,D=7,episode=row['key'],actual_policy_execution=False),indent=2))
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        peak=int(np.nanargmax(curve));indices=[max(0,peak-10),peak,min(len(curve)-1,peak+10)];fig=plt.figure(figsize=(9,5),layout='constrained');gs=fig.add_gridspec(2,3)
        for x,s in enumerate(indices):
            im=read_rgb(row,meta['names'][s+1]);cv2.imwrite(str(dest/f'{"ABC"[x]}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR));ax=fig.add_subplot(gs[0,x]);ax.imshow(im);ax.axis('off');ax.set_title(f'{"ABC"[x]}: recorded step {s+1}')
        ax=fig.add_subplot(gs[1,:]);ax.plot(np.arange(1,len(curve)+1),curve);ax.set(xlabel='Recorded action step',ylabel='Offline action entropy',title=f'G{ei:02d} — recorded demonstration')
        for s in indices:ax.axvline(s+1,ls=':',color='gray')
        for ext in ['png','pdf','svg']:fig.savefig(dest/f'G{ei:02d}.{ext}',dpi=180)
        plt.close(fig)
    (out/'complete.json').write_text(json.dumps(dict(time=time.time(),episodes=2,offline=True)))
if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
