"""Bounded read-only multi-run summary, scalar history and save evidence."""
import datetime,json,os,socket,subprocess
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
S=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003')
out={'time':datetime.datetime.now().astimezone().isoformat(),'runs':{}}
for name in ['formal-b16-v1','click-bell-v2','lift-pot-v1']:
 O=S/'runs'/name;row={'exists':O.exists(),'scalars':{},'receipts':{},'checkpoints':[]}
 for rel in ['final.json','error.json','recovery-error.json','state.json','startup_smoke/result.json']:
  p=O/rel
  if p.exists():
   value=json.loads(p.read_text())
   if rel=='final.json':value={k:value.get(k) for k in ['time','terminal_status','error','recovery_error','rlt_borrowed','rlt_return_dispatched']}
   row['receipts'][rel]=value
 for p in sorted((O/'formal/tensorboard/all').glob('events.out.tfevents.*')):
  e=EventAccumulator(str(p),size_guidance={'scalars':0});e.Reload()
  for tag in e.Tags()['scalars']:
   if tag in ['env/success_once','train/actor/grad_norm','time/step','time/generate_rollouts','time/actor_training','eval/success_once','eval/success_at_end']:
    values=e.Scalars(tag);row['scalars'][tag]={'count':len(values),'values':[{'step':v.step,'value':v.value,'wall_time':v.wall_time} for v in values]}
 for p in (O/'formal').glob('**/global_step_*'):
  if p.is_dir():row['checkpoints'].append({'path':str(p.relative_to(O)),'mtime':p.stat().st_mtime,'entries':sorted(c.name for c in p.iterdir())[:15]})
 out['runs'][name]=row
out['cloud_head']=subprocess.check_output(['git','-C',str(S/'publication/wmrl-bell-release-v1'),'ls-remote','--heads','personal','refs/heads/codex/wmrl-bell-reward-20261005'],text=True,timeout=90).strip()
print(json.dumps(out))
