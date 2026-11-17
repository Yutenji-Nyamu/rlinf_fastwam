"""Interleaved batched-K inference on one saved observation (NFE10)."""
import json,sys,time,os
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

def run(j):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from rlinf.models.embodiment.openpi import get_model
    from tools.appendix_signals.dojo_sampling import add_legacy_trace
    cfg=json.loads(Path(j['raw_config']).read_text())['model']
    model=get_model(OmegaConf.create(cfg)).cuda().eval().requires_grad_(False);add_legacy_trace(model)
    with np.load(j['inputs'],allow_pickle=False)as z:inp={k:z[k].copy()for k in z.files}
    packs={};noises={}
    for k in [1,2,3,4,8]:
        ob={a:torch.from_numpy(np.repeat(inp[a][:1],k,axis=0))for a in ['main_images','wrist_images','states']}
        ob.update(task_descriptions=inp['task_descriptions'][:1].tolist()*k,extra_view_images=None);packs[k]=ob
        noises[k]=torch.randn(k,50,32,generator=torch.Generator().manual_seed(1700)).cuda().to(model.action_in_proj.weight.dtype)
    def call(k):
        with torch.inference_mode():return model.predict_action_batch(packs[k],noise=noises[k],mode='eval',compute_values=False,num_steps=10)
    for _ in range(3):
        for k in packs:call(k)
    rows=[];rng=np.random.default_rng(171117);out=Path(j['output']);out.mkdir(parents=True,exist_ok=True)
    for repeat in range(30):
        for order,k in enumerate(rng.permutation([1,2,3,4,8]).tolist()):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();value=call(k);torch.cuda.synchronize()
            rows.append(dict(K=k,repeat=repeat,order=order,seconds=time.perf_counter()-t,allocated_peak_gib=torch.cuda.max_memory_allocated()/2**30));del value
    (out/'samples.json').write_text(json.dumps(rows,indent=2));summary=[]
    for k in packs:
        r=[x for x in rows if x['K']==k];v=np.array([x['seconds']for x in r])*1000
        summary.append(dict(K=k,median_ms=float(np.median(v)),q25_ms=float(np.quantile(v,.25)),q75_ms=float(np.quantile(v,.75)),peak_allocated_gib=max(x['allocated_peak_gib']for x in r)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'complete.json').write_text(json.dumps(dict(NFE=10,base_observations=1,repeats=30,warmups=3,order='interleaved shuffled',device=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(8,3),layout='constrained');v=np.array([r['median_ms']for r in summary]);axs[0].errorbar(list(packs),v,yerr=[v-[r['q25_ms']for r in summary],[r['q75_ms']for r in summary]-v],fmt='o-');axs[0].set(xlabel='K',ylabel='Latency (ms), median and IQR');axs[1].plot(list(packs),[r['peak_allocated_gib']for r in summary],'o-');axs[1].set(xlabel='K',ylabel='Peak allocated memory (GiB)')
    for ext in ['png','pdf','svg']:fig.savefig(out/f'batched-K.{ext}',dpi=180)
if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
