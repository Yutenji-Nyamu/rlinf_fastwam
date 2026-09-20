import importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from rlinf.algorithms.rlt.dvac_weighting import compute_endpoint_variances

def test_recording_preserves_actions_and_captures_clean_endpoints(monkeypatch):
 path=Path(__file__).parents[2]/'rlinf/models/embodiment/fastwam/fastwam_rl.py'
 spec=importlib.util.spec_from_file_location('tested_fastwam_flow',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 t=torch.tensor([1.,.7,.4,.1]);schedule=SimpleNamespace(timesteps=t*1000,normalized_timesteps=t,deltas=torch.tensor([-.3,-.3,-.3,-.1]))
 monkeypatch.setattr(m,'resolve_action_schedule',lambda *args:schedule)
 def velocity(model,*,x,raw_timestep,conditioning):return .2*x+raw_timestep/1000
 monkeypatch.setattr(m,'predict_action_velocity',velocity)
 model=SimpleNamespace(device='cpu',torch_dtype=torch.float32,infer_action_scheduler=SimpleNamespace(step=lambda v,dt,x:x+dt*v))
 x=torch.arange(2*32*14,dtype=torch.float32).reshape(2,32,14)/1000
 kw=dict(conditioning=None,initial_latents=x,num_inference_steps=4,sigma_shift=5,noise_level=.1,deterministic=True)
 clean=m.flow_sde_rollout(model,**kw);scored=m.flow_sde_rollout(model,collect_endpoint_previews=True,**kw)
 assert torch.equal(clean.actions,scored.actions) and clean.endpoint_previews is None
 expected=[];state=x.clone()
 for ti,dt in zip(t,schedule.deltas):
  v=.2*state+ti;expected.append(state-ti*v);state=state+dt*v
 assert torch.allclose(scored.endpoint_previews,torch.stack(expected,1),atol=1e-7)
 vv=compute_endpoint_variances(scored.endpoint_previews,l_values=(2,3,4))
 assert set(vv)=={2,3,4} and all(v.shape==(2,32) and torch.isfinite(v).all() for v in vv.values())
