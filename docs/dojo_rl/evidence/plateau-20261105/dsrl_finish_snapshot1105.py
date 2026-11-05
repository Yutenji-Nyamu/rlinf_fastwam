"""Read-only final DSRL scalar/occupancy snapshot; never creates a CUDA context."""
import os,json,time,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
roots={'rt_clean':'/data/chenyiteng/results/robotwin-dsrl-full-20261104/clean','dojo_clean':'/data/chenyiteng/results/dojo-offpolicy/dojo-dsrl-full-g5-20261104','dojo_u':'/data/chenyiteng/results/dojo-offpolicy/dojo-dsrl-u-g6-20261104','rt_u':'/data/chenyiteng/results/robotwin-dsrl-full-20261104/u-g7'}
out={'time':time.time(),'runs':{}}
for name,root in roots.items():
 p=next(Path(root).glob('formal/**/tensorboard'))
 a=EventAccumulator(str(p),size_guidance={'scalars':0});a.Reload()
 metrics={}
 for key in a.Tags()['scalars']:
  bystep={}
  for e in a.Scalars(key):
   if e.step not in bystep or bystep[e.step][2]<e.wall_time:bystep[e.step]=[e.step,e.value,e.wall_time]
  metrics[key]=[bystep[s] for s in sorted(bystep)]
 out['runs'][name]={'root':root,'metrics':metrics}
for key,cmd in {
 'gpu':'nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits',
 'disk':'df -BG /home /data',
 'memory':'free -h',
 'owners':'ps -p 3583941,3583943,3583954,3583955,3583990,3583991 -o pid,user,stat,etime,args'
}.items():out[key]=subprocess.getoutput(cmd)
print(json.dumps(out))
