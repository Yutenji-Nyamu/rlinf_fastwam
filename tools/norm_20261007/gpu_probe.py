"""Real-model Norm passivity probe on an explicitly leased single GPU."""
import argparse,json,os,random,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from rlinf.models.embodiment.openpi import get_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--role',choices=['bc','dsrl'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert a.gpu in (6,7) and os.environ['CUDA_VISIBLE_DEVICES']==str(a.gpu)
    assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
    cfg=OmegaConf.load(a.config);model=get_model(cfg.actor.model,torch_dtype=torch.bfloat16).to('cuda:0').eval()
    B=cfg.env.train.total_num_envs
    im=torch.arange(256*256*3).remainder(256).to(torch.uint8).reshape(1,256,256,3).expand(B,-1,-1,-1).clone()
    prompt=OmegaConf.select(cfg,'actor.model.openpi_data.default_prompt') or cfg.env.train.task_config.task_name.replace('_',' ')
    obs={'main_images':im,'wrist_images':torch.stack((im.flip(1),im.flip(2)),dim=1),'extra_view_images':None,'states':torch.linspace(-.25,.25,14).repeat(B,1),'task_descriptions':[prompt]*B}
    results=[];calls=[];original=model.sample_mean_var_val
    def counted(*args,**kwargs):calls.append(args[1]);return original(*args,**kwargs)
    model.sample_mean_var_val=counted
    for phase in ([0,1] if a.role=='dsrl' else [0]):
        if a.role=='dsrl':model.dsrl_policy_phase.fill_(phase);model.config.__dict__['dsrl_u_enabled']=False
        before=(torch.random.get_rng_state(),torch.cuda.get_rng_state(),random.getstate());calls.clear()
        mode='train' if a.role=='dsrl' else 'eval'
        with torch.no_grad():action0,info0=model.predict_action_batch(obs,mode=mode)
        after=(torch.random.get_rng_state(),torch.cuda.get_rng_state(),random.getstate())
        assert calls==list(range(10))
        torch.random.set_rng_state(before[0]);torch.cuda.set_rng_state(before[1]);random.setstate(before[2]);calls.clear()
        if a.role=='dsrl':model.config.__dict__['dsrl_u_enabled']=True
        with torch.no_grad():action1,info1=model.predict_action_batch(obs,mode=mode,**({'norm_enabled':True} if a.role=='bc' else {}))
        assert calls==list(range(10)),calls
        assert torch.equal(action0,action1),'environment action changed'
        for k in ('action','model_action','chains','denoise_inds'):
            if k in info0['forward_inputs']:assert torch.equal(info0['forward_inputs'][k],info1['forward_inputs'][k]),k
        assert torch.equal(after[0],torch.random.get_rng_state()) and torch.equal(after[1],torch.cuda.get_rng_state()) and after[2]==random.getstate()
        raw=info1['forward_inputs']['dsrl_u' if a.role=='dsrl' else 'norm_raw']
        assert raw.shape==(B,cfg.actor.model.num_action_chunks) and torch.isfinite(raw).all() and raw.min()>=0 and raw.std()>0
        assert all(not layer._forward_hooks for layer in model.paligemma_with_expert.gemma_expert.model.layers)
        results.append({'phase':phase,'batch':B,'shape':list(raw.shape),'min':raw.min().item(),'mean':raw.mean().item(),'max':raw.max().item(),'std':raw.std().item(),'main_action_exact':True,'cpu_cuda_python_rng_exact':True,'expert_forwards':len(calls),'hooks_removed':True})
    out={'status':'PASS','time':time.time(),'role':a.role,'gpu':a.gpu,'results':results,'peak_allocated':torch.cuda.max_memory_allocated()}
    Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
