#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
date --iso-8601=seconds
id
free -b
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
ps -u chenyiteng --sort=-rss -o pid,ppid,pgid,etime,stat,nlwp,rss,comm | head -n 9
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import json,re,datetime,csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
scalars={tag:[{'step':p.step+1,'value':p.value,'wall_time':p.wall_time} for p in ea.Scalars(tag)] for tag in ea.Tags()['scalars']}
log=(run/'runtime/driver.log').read_text(errors='replace')
resources=list(csv.DictReader((run/'runtime/resource.csv').open()))
ckpts=[]
for directory in run.glob('*/checkpoints/global_step_30'):
 for p in directory.rglob('*'):
  if p.is_file():ckpts.append({'path':str(p),'bytes':p.stat().st_size})
pid=int((run/'runtime/wrapper.pid').read_text())
out={'cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scalars':scalars,'resource':resources,'checkpoints':ckpts,'wrapper_alive':Path('/proc',str(pid)).exists(),'exit':(run/'runtime/exit_code.txt').read_text() if (run/'runtime/exit_code.txt').exists() else None,'last_progress':re.findall(r'Global Step:\s*\d+/100',log)[-1:],'last_rollout':[l for l in log.splitlines() if 'Generating Rollout Epochs:' in l][-1:],'fatal':{pat:len(re.findall(pat,log,re.I)) for pat in ('OIDN Error','pthread_key_create','Fatal Python error','Traceback','CUDA out of memory','OutOfMemoryError','non.?finite')}}
print('SIDNEY_JSON',json.dumps(out))
fast=Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2')
for name in ('exit_code.txt','finished_at.txt'):
 p=fast/'runtime'/name;print('FAST',name,p.read_text().strip())
for directory in fast.glob('*/checkpoints/global_step_30'):
 for p in directory.rglob('*'):
  if p.is_file():print('FAST_CKPT',p,p.stat().st_size)
PY
