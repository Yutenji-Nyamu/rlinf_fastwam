#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json,yaml,datetime,re,csv,subprocess,hashlib
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
choices={
 'fastwam-grpo':['fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1','fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2','fastwam-grpo-control-formal100-2gpu16x16-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-pi0style-v3','fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v1','fastwam-grpo-control-formal100-2gpu32x8-g8-b2048-u2-m10-fixed32-eval5-phys67-dcp-v2'],
 'grpo':['grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2','grpo-formal100-current-4gpu128train64eval-ppo-matched-v2'],
 'pi05':['pi05-grpo-control-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys45-localshard-v2','pi05-grpo-control-formal100-2gpu64x4-g8-b512-u5-m5-fixed32-eval5-phys45-localshard-v1'],
 'pi05-sidney':['move-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1','move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1']}
print('CST',datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),flush=True)
for family,names in choices.items():
 for name in names:
  root=base/family/'runs'/name; d={'family':family,'name':name,'path':str(root),'scalars':{},'checkpoints':[]}
  try:
   for p in root.glob('runtime/*resolved*.yaml'):d['config_path']=str(p);d['config']=yaml.safe_load(p.read_text())
   ea=EventAccumulator(str(root/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
   d['scalars']={tag:[{'step':p.step+1,'raw_step':p.step,'value':p.value,'wall_time':p.wall_time} for p in ea.Scalars(tag)] for tag in ea.Tags()['scalars']}
   logpath=root/'runtime/driver.log';log=logpath.read_text(errors='replace') if logpath.exists() else '';lines=log.splitlines()
   d['driver_bytes']=logpath.stat().st_size if logpath.exists() else 0
   d['last_progress']=re.findall(r'Global Step:\s*\d+/\d+',log)[-1:];d['last_rollout']=[l[-500:] for l in lines if 'Generating Rollout Epochs:' in l][-1:]
   d['fatal']={pat:{'count':sum(pat.lower() in l.lower() for l in lines),'first':[(i+1,l[:600]) for i,l in enumerate(lines) if pat.lower() in l.lower()][:2]} for pat in ('OIDN Error','pthread_key_create','Fatal Python error','Traceback','CUDA out of memory','OutOfMemoryError','non-finite')}
   d['trainable_logs']=[l[:1500] for l in lines if re.search(r'trainable|trainable_params|replay.*parity|replay.*logprob',l,re.I)][:30]
   d['runtime']={p.name:p.read_text().strip() for p in [root/'runtime'/n for n in ('wrapper.pid','exit_code.txt','finished_at.txt','started_at.txt')] if p.exists()}
   pid=d['runtime'].get('wrapper.pid');d['wrapper_alive']=bool(pid and Path('/proc',pid).exists())
   rp=root/'runtime/resource.csv';d['resource']=list(csv.DictReader(rp.open())) if rp.exists() else []
   for p in root.glob('*/checkpoints/global_step_*/actor/**/*'):
    if p.is_file():d['checkpoints'].append({'path':str(p),'bytes':p.stat().st_size})
  except Exception as e:d['error']=repr(e)
  print('RUN_JSON',json.dumps(d),flush=True)
wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
for name in ('fastwam-current-grpo','grpo-pi0-robotwin-7d07a421','pi05-robotwin-rl','sidney-pi05-current-rlinf'):
 p=wt/name;print('WORKTREE',p)
 for args in (['rev-parse','HEAD'],['status','--short']):print(subprocess.run(['git','--no-optional-locks','-C',str(p),*args],capture_output=True,text=True).stdout)
 for folder in ('rlinf/models','rlinf/algorithms','rlinf/workers/actor'):
  files=subprocess.run(['rg','--files',str(p/folder)],capture_output=True,text=True).stdout.splitlines()
  print('SOURCE_FILES',name,[f for f in files if any(x in f for x in ('fastwam','openpi','grpo','advantages','actor_loss','fsdp_actor','losses'))])
PY
