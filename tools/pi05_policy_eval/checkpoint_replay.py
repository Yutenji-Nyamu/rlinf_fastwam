"""Reuse saved final-policy observations/noises at a single GRPO checkpoint."""
import json,os,sys,time
from pathlib import Path
import numpy as np
from entropy_analysis import entropy,reached_positions,executed_count
from teacher_replay import load_observation

def main(path):
 c=json.loads(Path(path).read_text());source=Path(c['model_source'])
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(c['gpu'])
 sys.path.insert(0,str(source));sys.path.insert(0,str(source/'tools/pi05_u_video'))
 from noise_samples import numpy_exact,repeated_observation
 import torch
 from omegaconf import OmegaConf
 from rlinf.models.embodiment.openpi import get_model
 torch.set_num_threads(1);torch.cuda.set_device(0);torch.backends.cuda.enable_cudnn_sdp(False)
 batch=Path(c['input_batch']);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
 original=json.loads((batch/'config.json').read_text())
 model=get_model(OmegaConf.create(original['model']))
 if c.get('weights'):model.load_state_dict(torch.load(c['weights'],map_location='cpu',weights_only=True),strict=True)
 model=model.cuda().eval();episodes,reached=reached_positions(batch);rows=[]
 def infer(obs,noise):
  with torch.inference_mode():
   _,r=model.predict_action_batch(obs,mode='eval',compute_values=False,num_steps=10,noise=torch.from_numpy(noise).cuda(),return_dvac_telemetry=True)
   t=r['dvac_telemetry'];np.testing.assert_array_equal(noise,numpy_exact(t['initial_noise_full']))
   return numpy_exact(t['final_model_action'])
 for f in sorted(batch.glob('query_*.npz')):
  if f.stem.endswith('_obs'):continue
  query=int(f.stem.split('_')[-1]);obs=load_observation(batch/f'query_{query:03d}_obs.npz')
  with np.load(f,allow_pickle=False) as z:
   noise=z['noise4_initial_noise'].copy();b=len(noise)
   main=infer(obs,noise[:,0])
   if c['checkpoint_step']==200:
    np.testing.assert_array_equal(main,z['noise4_model_actions'][:,0])
   extra=infer(repeated_observation(obs,3),np.concatenate([noise[:,i]for i in [1,2,3]]))
   actions=np.stack([main,*extra.reshape(3,b,50,14)],axis=1)
   h=entropy(actions)
   for e in episodes:
    s=e['slot'];n=executed_count(reached,s,query,z['before'][s],z['after'][s],z['active'][s])
    for j in range(n):rows.append({'pool_batch':c['pool_batch'],'slot':s,'seed':e['actual'],'query':query,'h':j,'success':e['success'],'entropy':float(h[s,j])})
  np.savez_compressed(out/f.name,model_actions=actions,initial_noise=noise)
 (out/'positions.json').write_text(json.dumps(rows))
 (out/'receipt.json').write_text(json.dumps({'checkpoint_step':c['checkpoint_step'],'positions':len(rows),'queries':len(list(out.glob('query_*.npz'))),'gpu':c['gpu'],'peak_gib':torch.cuda.max_memory_allocated()/2**30,'exact_final_main':c['checkpoint_step']==200,'completed':time.time()}))

if __name__=='__main__':main(sys.argv[1])
