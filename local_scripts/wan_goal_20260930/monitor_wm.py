"""Read-only progress snapshot for this one WM sequence."""
import argparse,json,os,re,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.modules.setdefault('tensorflow',None)
parser=argparse.ArgumentParser();parser.add_argument('--run',default='wan-goal-sz3-20261001-r2')
args=parser.parse_args();assert re.fullmatch(r'wan-goal-sz3-[A-Za-z0-9-]+',args.run)
root=Path('/data/chenyiteng/projects/wan-goal-sz3/runs')/args.run
out={'time':time.time(),'files':{},'stages':{}}
for f in root.glob('*current.json'):out['files'][f.name]=json.loads(f.read_text())
for name in ('oft-smoke','pi05-smoke','pi05-formal'):
 run=root/name
 if not run.exists():continue
 row={'exit':None,'scalars':{},'checkpoint_files':[]};out['stages'][name]=row
 f=run/'wm-exit.json'
 if f.exists():row['exit']=json.loads(f.read_text())
 f=root/(name+'-control')/'smoke-verification.json'
 if f.exists():row['verification']=json.loads(f.read_text())
 f=run/'command.log'
 if f.exists():
  with f.open('rb') as stream:stream.seek(max(0,f.stat().st_size-6000));tail=stream.read().decode(errors='replace')
  row['log']={'mtime':f.stat().st_mtime,'bytes':f.stat().st_size,'tail':re.sub(r'\x1b\[[0-9;]*m','',tail)[-1600:]}
 for d in (run/'tensorboard/all',run/'tensorboard'):
  if not list(d.glob('events.out.tfevents.*')):continue
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  reader=EventAccumulator(str(d),size_guidance={'scalars':0});reader.Reload()
  for tag in reader.Tags().get('scalars',[]):
   if any(k in tag for k in ('grad_norm','loss_mask_fraction','advantages_','total_loss','episode_return','success','time/total')):
    row['scalars'][tag]=[{'step':x.step,'value':x.value,'wall_time':x.wall_time} for x in reader.Scalars(tag)[-3:]]
  break
 for pattern in ('*/checkpoints/global_step_*/actor/dcp_checkpoint/*','*/checkpoints/global_step_*/actor/model_state_dict/*'):
  for f in run.glob(pattern):
   if f.is_file():row['checkpoint_files'].append({'file':str(f.relative_to(run)),'size':f.stat().st_size,'mtime':f.stat().st_mtime})
print(json.dumps(out,ensure_ascii=False))
