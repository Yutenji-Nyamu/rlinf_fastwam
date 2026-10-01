"""One read-only CPU check of the original first formal checkpoint; no CUDA/model launch."""
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.modules.setdefault('tensorflow',None)
R=Path('/data/chenyiteng/projects/wan-goal-sz3')
F=R/'runs/wan-goal-sz3-20261001-r6/pi05-formal'
sys.path.insert(0,str(R/'scripts'))
from verify_smoke import checkpoint_structure
experiment='wan_goal_pi05_headonly_formal_sz3'
destination=F.parent/'checkpoint40-verification.json'
assert os.getuid()==20001 and not destination.exists()
structure=checkpoint_structure(F.resolve(strict=True),experiment,40)
weights=Path(structure['full_weights'])
files=[weights,Path(structure['actor_dir'])/'dcp_checkpoint/.metadata',
       *map(Path,structure['storage_files'])]
def snapshot():
    result={}
    for path in files:
        stat=path.stat()
        result[str(path)]=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
    return result
before=snapshot()
time.sleep(2)
assert before==snapshot(), 'Checkpoint writer still active; retry later'
assert structure==checkpoint_structure(F.resolve(strict=True),experiment,40), 'Checkpoint metadata changed; retry later'
import torch
torch.set_num_threads(8)
state=torch.load(weights,weights_only=True,mmap=True,map_location='cpu')
assert isinstance(state,dict)
floating_tensors=0;floating_elements=0;bad=[]
for name,tensor in state.items():
    if isinstance(tensor,torch.Tensor) and tensor.is_floating_point():
        floating_tensors+=1;floating_elements+=tensor.numel()
        if not bool(torch.isfinite(tensor).all()):bad.append(name)
assert floating_tensors>0 and not bad, {'nonfinite_parameter_names':bad[:10]}
assert not torch.cuda.is_initialized(), 'CPU-only verification initialized CUDA'
assert before==snapshot(), 'Checkpoint changed during verification'
report={'time':time.time(),'ok':True,'step':40,'checkpoint_structure':structure,
        'floating_parameters':{'tensors':floating_tensors,'elements':floating_elements,'all_finite':True},
        'cuda_initialized':False,'read_only':True,'stable_checkpoint_files':len(files),
        'scope':'Own generated DCP metadata/storage extents and all floating full-model parameters; no optimizer restore or real LIBERO evaluation'}
with destination.open('x') as out:json.dump(report,out,indent=2);out.write('\n')
print(json.dumps(report),flush=True)
