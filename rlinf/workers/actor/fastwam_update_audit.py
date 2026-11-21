"""Record the first actual action-expert update on each FSDP rank."""
import json, os, time
from pathlib import Path
import torch

def observe_update(worker):
    if getattr(worker, "_fastwam_step_audited", False):
        return worker.optimizer_step()
    optimizer = worker.optimizer
    picks=[]
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None: continue
            x=p.to_local() if hasattr(p,"to_local") else p
            grad=p.grad.to_local() if hasattr(p.grad,"to_local") else p.grad
            if x.numel() and grad.numel():
                if x.dtype != torch.float32: raise RuntimeError("FastWAM master shard must be float32")
                n=min(512,x.numel())
                picks.append((p,x.detach().reshape(-1)[:n].clone()))
                if len(picks)==12:break
        if len(picks)==12:break
    result=worker.optimizer_step()
    changed=0; peak=0.; moments=set()
    for p,before in picks:
        x=p.to_local() if hasattr(p,"to_local") else p
        diff=(x.detach().reshape(-1)[:before.numel()]-before).abs()
        peak=max(peak,diff.max().item());changed+=int(diff.count_nonzero())
        state=optimizer.state.get(p,{})
        for key in ('exp_avg','exp_avg_sq'):
            if key in state:moments.add(str(state[key].dtype))
    out=dict(time=time.time(),rank=os.environ['RANK'],observed_tensors=len(picks),changed_coordinates=changed,max_delta=peak,master='torch.float32',moments=sorted(moments))
    d=Path(os.environ['DOJO_GRPO_OWNER'])
    (d/f"first-update-rank-{os.environ['RANK']}.json").write_text(json.dumps(out))
    if picks and changed:
        worker._fastwam_step_audited=True
    return result

