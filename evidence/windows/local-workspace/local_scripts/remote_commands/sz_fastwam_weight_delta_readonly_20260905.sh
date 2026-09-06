#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,io,json,os,pickle
from pathlib import Path
import torch
assert os.getuid()==1003
torch.set_num_threads(2)
root=Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3')
ckpt=next(root.glob('*/checkpoints/global_step_10/actor/dcp_checkpoint'))
with (ckpt/'.metadata').open('rb') as f:meta=pickle.load(f)
source=Path('/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt')
payload=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
keys=['mixtures.action.action_encoder.weight','mixtures.action.head.weight','mixtures.action.time_embedding.0.weight','mixtures.action.blocks.0.self_attn.q.weight','mixtures.action.blocks.29.self_attn.q.weight','mixtures.action.blocks.0.modulation']
read_bytes=0
def dcp_tensor(key):
    global read_bytes
    md=meta.state_dict_metadata[key]
    value=torch.empty(tuple(md.size),dtype=md.properties.dtype)
    pieces=[]
    for idx,info in meta.storage_data.items():
        if idx.fqn!=key:continue
        with (ckpt/info.relative_path).open('rb') as f:
            f.seek(info.offset);raw=f.read(info.length)
        read_bytes+=len(raw)
        part=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        if value.ndim:
            off=idx.offset
            slices=tuple(slice(int(o),int(o)+int(n)) for o,n in zip(off,part.shape))
            value[slices]=part
        else:value.copy_(part)
        pieces.append({'offset':list(idx.offset or []),'shape':list(part.shape),'bytes':info.length})
    if not pieces:raise RuntimeError('No chunks: '+key)
    return value,pieces
out={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'source':str(source),'checkpoint':str(ckpt),'source_top_keys':list(payload),'selected':[],'gpu_model_run':False,'optimizer_run':False}
for key in keys:
    model_key='fsdp_checkpoint.model.model.mot.'+key
    optim_prefix='fsdp_checkpoint.optimizers.state.model.mot.'+key
    current,pieces=dcp_tensor(model_key)
    initial=payload['mot'][key].to(dtype=current.dtype)
    assert initial.shape==current.shape
    m,_=dcp_tensor(optim_prefix+'.exp_avg');v,_=dcp_tensor(optim_prefix+'.exp_avg_sq');step,_=dcp_tensor(optim_prefix+'.step')
    a=initial.float();b=current.float();delta=b-a;unchanged=current==initial
    row={'key':key,'shape':list(current.shape),'numel':current.numel(),'dtype':str(current.dtype),'source_dtype':str(payload['mot'][key].dtype),'optimizer_step':step.item(),
         'changed_fraction':float((~unchanged).float().mean()),'relative_l2':float(torch.linalg.vector_norm(delta)/(torch.linalg.vector_norm(a)+1e-30)),
         'max_abs_delta':float(delta.abs().max()),'mean_abs_weight':float(a.abs().mean()),'mean_abs_delta':float(delta.abs().mean()),
         'moment_nonzero_fraction':float((m!=0).float().mean()),'unchanged_with_nonzero_moment_fraction':float((unchanged & (m!=0)).float().mean()),
         'moment_mean_abs':float(m.float().abs().mean()),'pieces':pieces}
    out['selected'].append(row)
out['dcp_bytes_read']=read_bytes
print(json.dumps(out))
PY
