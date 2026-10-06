import datetime,json,os,socket,zipfile
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003');O=S/'runs/lift-pot-v1'
cfg=json.loads((S/'lift-pot-v1/prepared/formal.yaml').read_text())
out={'time':datetime.datetime.now().astimezone().isoformat(),'parallel':{'placement':cfg['cluster']['component_placement'],'train':{k:cfg['env']['train'][k] for k in ['total_num_envs','rollout_epoch','group_size','chunk','max_episode_steps']},'actor':{k:cfg['actor'][k] for k in ['global_batch_size','micro_batch_size']},'update_epoch':cfg['algorithm']['update_epoch']},'checkpoint':[],'bottle':{}}
cp=O/'formal/lift-pot-v1-formal/checkpoints/global_step_10'
for p in cp.rglob('*'):
 if p.is_file():
  row={'path':str(p.relative_to(cp)),'bytes':p.stat().st_size}
  if p.name.endswith(('.pt','.pth')):row['readable_zip_directory']=zipfile.is_zipfile(p)
  out['checkpoint'].append(row)
for p in (S/'runs/formal-b16-v1/formal').glob('**/tensorboard/all/events.out.tfevents.*'):
 e=EventAccumulator(str(p),size_guidance={'scalars':0});e.Reload()
 for tag in ['eval/success_once','env/success_once','train/actor/grad_norm']:
  if tag in e.Tags()['scalars']:
   rows=e.Scalars(tag);out['bottle'][tag]={'count':len(rows),'last':[{'step':v.step,'value':v.value} for v in rows[-7:]]}
print(json.dumps(out))
