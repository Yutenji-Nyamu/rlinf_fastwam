"""Shared logging and checkpoint helpers from the clean EXPO driver."""
import hashlib
import json
import time

import torch

def log(run,event,**fields):
    row={'time':time.time(),'event':event,**fields}
    with (run/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def digest(value):
    h=hashlib.sha256()
    def feed(x):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().contiguous()
            h.update(str((tuple(x.shape),x.dtype)).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());feed(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:feed(v)
        else:h.update(repr(x).encode())
    feed(value);return h.hexdigest()

def check_finite(value):
    if isinstance(value,torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():raise FloatingPointError('Nonfinite checkpoint tensor')
    elif isinstance(value,dict):
        for v in value.values():check_finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:check_finite(v)


def frozen_sample(backend):
    return digest({n:p.detach().reshape(-1)[:64] for n,p in backend.model.named_parameters() if n not in backend.trainable})

