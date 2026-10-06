"""CPU load the real task reward and score the prepared initial main images."""
import json,os,socket,sys,time
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');D=S/'lift-pot-v1'
sys.path.insert(0,str(D/'code'))
import numpy as np
import torch
from rm_adapter import LiftPotReward,SUCCESS_THRESHOLD
torch.set_num_threads(4)
begin=time.time();model=LiftPotReward(S/'task-reward-v2/lift-pot/best.pt')
with np.load(D/'assets/lift_pot_clean50_reset.npz',allow_pickle=False) as data:
 images=data['main_images'];ids=data['reset_ids'].tolist()
scores=[]
for start in range(0,len(images),16):scores.extend(model.compute_reward(torch.from_numpy(images[start:start+16])).tolist())
out=dict(cpu_only=True,checkpoint=model.checkpoint_sha256,threshold=SUCCESS_THRESHOLD,count=len(scores),
         min=min(scores),median=float(np.median(scores)),max=max(scores),positive_count=sum(v>=SUCCESS_THRESHOLD for v in scores),
         scores=[dict(reset_id=k,score=v) for k,v in zip(ids,scores)],elapsed_seconds=time.time()-begin,cuda_initialized=torch.cuda.is_initialized())
assert not out['cuda_initialized']
path=D/'assets/initial_reward_scores.json'
assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
