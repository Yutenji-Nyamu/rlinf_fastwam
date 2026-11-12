"""K4 at every saved observation; no new environment rollout or interpolation."""
import argparse,copy,json,gc,time
from pathlib import Path
import numpy as np
from tools.appendix_signals.common import cuda_guard,entropy,write_json,write_npz
from tools.appendix_signals.dojo_sampling import add_legacy_trace

def main(args):
 torch=cuda_guard(args.gpu)
 from omegaconf import OmegaConf
 from rlinf.models.embodiment.openpi import get_model
 from tools.pi05_u_video.noise_samples import numpy_exact
 root=Path(args.data);out=Path(args.output)/'dojo-dense';allmeta=[]
 for task,label in [('put_bottles_into_dustbin','collect-dustbin'),('pour_liquid_into_cup','collect-pour')]:
  config=OmegaConf.load(Path(args.project)/'deployment-retry'/label/'config/formal.yaml')
  mc=copy.deepcopy(config.rollout.get('rlt_feature_model',config.actor.model));mc.openpi.use_dsrl=False
  if 'use_rlt' in mc.openpi:mc.openpi.use_rlt=False
  model=get_model(mc).cuda().eval().requires_grad_(False);add_legacy_trace(model)
  folder=root/'dojo'/task;dest=out/task;queries=[];timings=[]
  for path in sorted((folder/'observations').glob('q*.npz')):
   q=int(path.stem[1:]);z=np.load(path,allow_pickle=False)
   active=np.asarray(z['active']) & (z['after']>z['before']);idx=np.flatnonzero(active)
   if not len(idx):continue
   obs={}
   for key in z.files:
    if key in ['before','after','active','success']:continue
    arr=z[key][idx]
    obs[key]=arr.tolist() if arr.dtype.kind in 'US' else torch.from_numpy(arr.copy())
   obs.setdefault('extra_view_images',None);samples=[];start=time.monotonic()
   for k in range(4):
    torch.manual_seed(111200000+q*4+k);torch.cuda.manual_seed_all(111200000+q*4+k)
    with torch.inference_mode():
     _,ret=model.predict_action_batch(obs,mode='eval',compute_values=False,num_steps=10,return_dvac_telemetry=True,record_signals=False)
    samples.append(numpy_exact(ret['dvac_telemetry']['final_model_action']))
   a=np.stack(samples,axis=1);h=entropy(a);pathout=dest/'queries'/path.name
   write_npz(pathout,model_actions=a,entropy=h,slots=idx,before=z['before'][idx],after=z['after'][idx],success=z['success'][idx])
   row={'query':q,'path':'queries/'+path.name,'slots':idx.tolist(),'K':4,'H':a.shape[2],'D':a.shape[3],'seconds':time.monotonic()-start}
   queries.append(row);write_json(dest/'queries.json',queries);print(task,q,len(idx),flush=True)
  eps=json.loads((folder/'episodes.json').read_text());write_json(dest/'episodes.json',eps)
  meta={'task':task,'queries':len(queries),'episodes':len(eps),'K':4,'mode':'independent offline K4 on every reached saved observation','no_interpolation':True,'no_new_rollout':True,'elapsed_inference_s':sum(x['seconds'] for x in queries)}
  write_json(dest/'complete.json',meta);allmeta.append(meta);del model;gc.collect();torch.cuda.empty_cache()
 write_json(out/'complete.json',allmeta)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--project',required=True);p.add_argument('--gpu',type=int,default=4);main(p.parse_args())
