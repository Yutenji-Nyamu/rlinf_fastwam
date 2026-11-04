"""One frozen-model observation probe; no environment or training mutation."""
import os,json,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.models.model import Observation
from rlinf.models.embodiment.openpi import get_model
torch.set_num_threads(4)
D=Path('/data/chenyiteng/deployment-20261103/dojo-offpolicy-g6-r3-n8-recovery-1')
raw=np.load('/data/chenyiteng/results/dojo-offpolicy/diagnosis-20261104/initial-observation.npz')
obs={k:torch.from_numpy(raw[k][:8]) for k in ('states','main_images','wrist_images')}
obs.update(task_descriptions=[str(s)for s in raw['task_descriptions'][:8]],extra_view_images=None)
started=time.monotonic();model=get_model(OmegaConf.load(D/'config/formal.yaml').actor.model).eval().cuda()
processed=model.input_transform(model.obs_processor(obs),transpose=False)
observation=Observation.from_dict(model.precision_processor(processed))
torch.manual_seed(1104);z=torch.randn(8,32,device='cuda',dtype=torch.float32)
repeated=z[:,None,:].repeat(1,50,1).to(next(model.dsrl_action_noise_net.parameters()).dtype)
g=torch.Generator(device='cuda');g.manual_seed(1104);iid=torch.randn(8,50,32,device='cuda',generator=g);iid[:,0]=z
def f(noise):
 out=model.sample_actions(observation,noise=noise,mode='eval',compute_values=False)
 return model.output_transform({'actions':out['actions'],'state':observation.state})['actions'].float().cpu().numpy()
with torch.inference_mode():
 full=f(iid);same=f(repeated)
 g.manual_seed(2104);iid2=torch.randn(8,50,32,device='cuda',generator=g);iid2[:,0]=z
 independent_control=f(iid2)
 same_precision=f(iid.to(repeated.dtype))
 torch.manual_seed(1104)
 actual,_=model.predict_action_batch(obs,mode='eval',compute_values=False)
 actual=actual.float().cpu().numpy()
def dist(a,b):
 delta=a-b
 return {'rmse_all14':float(np.sqrt(np.mean(delta**2))),'rmse_arm12':float(np.sqrt(np.mean(delta[:,:,[0,1,2,3,4,5,7,8,9,10,11,12]]**2))),'max_abs':float(np.abs(delta).max()),'per_env_rmse':[float(np.sqrt(np.mean(v**2)))for v in delta]}
report={'time':time.time(),'seconds':time.monotonic()-started,'observations':8,'policy_phase':int(model.dsrl_policy_phase.item()),'iid_vs_repeated':dist(full,same),'iid_vs_iid_control':dist(full,independent_control),'iid_dtype_control':dist(full,same_precision),'matched_dtype_iid_vs_repeat':dist(same_precision,same),'actual_dsrl_vs_expected_repeat':dist(actual,same),'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'note':'Frozen original model, saved observations; no closed-loop success measurement.'}
out=Path('/data/chenyiteng/results/dojo-offpolicy/diagnosis-20261104');out.mkdir(exist_ok=True)
np.savez_compressed(out/'noise-shadow-actions.npz',independent_actions=full,repeated_actions=same,actual_dsrl_actions=actual,initial_states=raw['states'][:8],independent_control=independent_control,independent_bfloat16=same_precision)
(out/'noise-shadow.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
