"""Append four independent samples to one recorded K4 trajectory.

Reuses the original OpenPI telemetry sampler and appendix entropy/cost definitions.
"""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np


def save(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2))


def entropy(a):
    # tools.appendix_signals.common.entropy: normalized D14, ddof=1, eps=1e-6.
    a=np.asarray(a,dtype=np.float64)
    return .5*np.log(2*np.pi*np.e*(a.var(axis=1,ddof=1)+1e-6)).sum(-1)


def curves(samples,positions,queries):
    index={q:i for i,q in enumerate(queries)}
    result={}
    for k in [2,3,4,8]:
        if k>samples.shape[1]:continue
        e=entropy(samples[:,:k])
        result[f'entropy{k}']=np.array([e[index[p['query']],p['h']]for p in positions])
    return result


def prepare(job):
    raw=Path(job['raw']);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    series=json.loads(Path(job['series']).read_text());pos=series['positions'];slot=series['metadata']['slot']
    queries=sorted({p['query']for p in pos});qm=json.loads((raw/'queries.json').read_text());qm=qm.get('queries',qm)if isinstance(qm,dict)else qm
    records={r['query']:r for r in qm};obs={};old=[];oldenv=[];noise=[];descriptions=[]
    for q in queries:
        with np.load(raw/f'query_{q:03d}_obs.npz',allow_pickle=False)as a:
            for name in a.files:obs.setdefault(name,[]).append(a[name][slot].copy())
        with np.load(raw/f'query_{q:03d}.npz',allow_pickle=False)as a:
            old.append(a['noise4_model_actions'][slot].copy());oldenv.append(a['noise4_env_actions'][slot].copy());noise.append(a['noise4_initial_noise'][slot].copy())
        descriptions.append(records[q]['descriptions'][slot])
    obs={k:np.stack(v)for k,v in obs.items()};old=np.stack(old);oldenv=np.stack(oldenv);noise=np.stack(noise)
    np.savez_compressed(out/'inputs.npz',**obs,old_model_actions=old,old_env_actions=oldenv,old_initial_noise=noise,
                        task_descriptions=np.asarray(descriptions),query_ids=np.asarray(queries))
    for k,v in curves(old,pos,queries).items():np.testing.assert_allclose(v,[p[k]for p in pos],rtol=0,atol=1e-8)
    save(out/'series-original.json',series)
    fm=json.loads((raw/series['metadata']['frame_map']).read_text());save(out/'frame-map.json',fm)
    import cv2
    cap=cv2.VideoCapture(str(raw/series['metadata']['video']));selected=[]
    for label,action,sec in [('A',184,7.40),('B',247,9.88),('C',316,12.76)]:
        frame=int(round(sec*fm['fps']));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read()
        if not ok:raise ValueError(f'Frame {frame} missing')
        cv2.imwrite(str(out/f'frame-{label}.png'),img)
        selected.append(dict(label=label,action=action,video_seconds=sec,frame=frame,frame_metadata=next(x for x in fm['frames']if x['frame']==frame)))
    cap.release();save(out/'selected-frames.json',selected)
    save(out/'prepared.json',dict(observations=len(queries),new_samples=len(queries)*4,action_points=len(pos),old_curves_match=True))
    return series,queries


def render(out,arrays,cost):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    marks=json.loads((out/'selected-frames.json').read_text())
    fig=plt.figure(figsize=(10,5.3),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1,1.25])
    for i,m in enumerate(marks):
        ax=fig.add_subplot(gs[0,i]);ax.imshow(Image.open(out/f"frame-{m['label']}.png"));ax.axis('off');ax.set_title(f"{m['label']}   Action {m['action']}",fontsize=10)
    ax=fig.add_subplot(gs[1,:]);colors={2:'#b5b5b5',3:'#df9e42',4:'#2380bb',8:'#303642'}
    for k in [2,3,4,8]:ax.plot(arrays['action'],arrays[f'entropy{k}'],color=colors[k],lw=1.0 if k<4 else 1.5,alpha=.85,label=f'K = {k}')
    for m in marks:ax.axvline(m['action'],ls=':',color='#555555',lw=.8);ax.text(m['action'],.99,m['label'],transform=ax.get_xaxis_transform(),ha='center',va='top')
    ax.set(xlabel='Action step',ylabel='Action entropy',xlim=(0,len(arrays['action'])-1));ax.spines[['top','right']].set_visible(False);ax.legend(ncols=4,frameon=False,loc='lower left');ax.grid(axis='y',alpha=.15)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'K-N17.{ext}',dpi=200)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.4,2.7),layout='constrained')
    for kind,label in [('batched_K','Batched generation'),('preserved_main_then_extras','Main then extra samples')]:
        ks=sorted({r['K']for r in cost if r['kind']==kind})
        axs[0].plot(ks,[np.median([r['seconds']for r in cost if r['kind']==kind and r['K']==k])*1000 for k in ks],'-o',label=label,ms=4)
    ks=[1,2,3,4,8];axs[1].plot(ks,[max(r['allocated_peak_gib']for r in cost if r['kind']=='batched_K'and r['K']==k)for k in ks],'-o',color='#2380bb',ms=4)
    axs[0].set(xlabel='Number of samples K',ylabel='Inference latency (ms)');axs[0].legend(frameon=False,fontsize=8)
    axs[1].set(xlabel='Number of samples K',ylabel='Peak allocated memory (GiB)')
    for ax in axs:ax.set_xticks(ks);ax.spines[['top','right']].set_visible(False)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'K-cost.{ext}',dpi=200)
    plt.close(fig)


def run(job):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    sys.path.insert(0,job['model_source'])
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    from tools.pi05_u_video.noise_samples import numpy_exact
    out=Path(job['output']);raw=Path(job['raw']);series=json.loads((out/'series-original.json').read_text());cfg=json.loads((raw/'config.json').read_text())
    model=get_model(OmegaConf.create(cfg['model'])).cuda().eval();model.requires_grad_(False);torch.backends.cuda.enable_cudnn_sdp(False)
    with np.load(out/'inputs.npz',allow_pickle=False)as a:inputs={k:a[k].copy()for k in a.files}
    qids=inputs['query_ids'].tolist();obskeys=[k for k in inputs if k not in ['old_model_actions','old_env_actions','old_initial_noise','task_descriptions','query_ids']]
    def pack(ids):
        ob={k:torch.from_numpy(inputs[k][ids].copy())for k in obskeys};ob['task_descriptions']=inputs['task_descriptions'][ids].tolist();ob['extra_view_images']=None;return ob
    def predict(ob,noise):
        with torch.inference_mode():
            ae,r=model.predict_action_batch(ob,noise=noise.cuda().to(model.action_in_proj.weight.dtype),mode='eval',compute_values=False,num_steps=10,return_dvac_telemetry=True,record_signals=False)
        return numpy_exact(r['dvac_telemetry']['final_model_action']),numpy_exact(ae)
    # Reuse the original native B16 replay check, once, before adding samples.
    with np.load(raw/'query_000_obs.npz',allow_pickle=False)as a:ob={k:torch.from_numpy(a[k].copy())for k in a.files}
    qm=json.loads((raw/'queries.json').read_text());qm=qm.get('queries',qm)if isinstance(qm,dict)else qm
    ob.update(task_descriptions=qm[0]['descriptions'],extra_view_images=None)
    with np.load(raw/'query_000.npz',allow_pickle=False)as a:replay_noise=a['noise4_initial_noise'][:,0].copy();expected=a['noise4_model_actions'][:,0].copy()
    actual,_=predict(ob,torch.from_numpy(replay_noise));delta=float(np.abs(actual-expected).max());save(out/'replay-check.json',dict(max_abs=delta));np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-5)
    seeds=np.array([[171117000+q*4+s for s in range(4)]for q in qids]);noise=torch.stack([torch.randn(50,32,generator=torch.Generator().manual_seed(int(s)))for s in seeds.flat])
    ids=np.repeat(np.arange(len(qids)),4);new,physical=predict(pack(ids),noise)
    new=new.reshape(len(qids),4,50,14);physical=physical.reshape(len(qids),4,50,14);all_samples=np.concatenate([inputs['old_model_actions'],new],axis=1)
    arrays=curves(all_samples,series['positions'],qids)
    arrays.update(action=np.array([p['action']for p in series['positions']]),query=np.array([p['query']for p in series['positions']]),h=np.array([p['h']for p in series['positions']]))
    np.savez_compressed(out/'samples-K8.npz',model_actions=all_samples,env_actions=np.concatenate([inputs['old_env_actions'],physical],axis=1),initial_noise=np.concatenate([inputs['old_initial_noise'],noise.numpy().reshape(len(qids),4,50,32)],axis=1),new_noise_seeds=seeds,query_ids=np.array(qids))
    np.savez_compressed(out/'action-entropy.npz',**arrays)
    import csv
    with(out/'action-entropy.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(arrays));w.writeheader();w.writerows({k:v[i]for k,v in arrays.items()}for i in range(len(arrays['action'])))
    cost=[]
    for k in [1,2,3,4,8]:
        n=torch.randn(k,50,32,generator=torch.Generator().manual_seed(1900+k));ob=pack(np.zeros(k,dtype=int));predict(ob,n)
        for kind in ['batched_K']+(['preserved_main_then_extras']if k>1 else []):
            for repeat in range(5):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                if kind=='batched_K':predict(ob,n)
                else:predict(pack(np.zeros(1,dtype=int)),n[:1]);predict(pack(np.zeros(k-1,dtype=int)),n[1:])
                torch.cuda.synchronize();cost.append(dict(kind=kind,K=k,B=1,repeat=repeat,seconds=time.perf_counter()-start,allocated_peak_gib=torch.cuda.max_memory_allocated()/2**30,reserved_peak_gib=torch.cuda.max_memory_reserved()/2**30))
    save(out/'costs.json',cost);save(out/'device.json',dict(name=torch.cuda.get_device_name(0),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,NFE=10,H=50,seed=job.get('noise_seed',171117000)))
    render(out,arrays,cost)
    save(out/'complete.json',dict(time=time.time(),observations=len(qids),new_samples=len(qids)*4,action_points=len(arrays['action']),Ks=[2,3,4,8],timed_Ks=[1,2,3,4,8],K4_K8_correlation=float(np.corrcoef(arrays['entropy4'],arrays['entropy8'])[0,1])))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args();job=json.loads(Path(args.job).read_text())
    if args.prepare_only:print(prepare(job)[1])
    else:run(job)
