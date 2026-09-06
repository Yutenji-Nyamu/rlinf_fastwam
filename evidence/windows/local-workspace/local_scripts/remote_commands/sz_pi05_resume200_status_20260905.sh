#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,json,re,subprocess
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
rt=run/'runtime-resume100-to200'
tz=datetime.timezone(datetime.timedelta(hours=8))
state={k:(rt/k).read_text().strip() if (rt/k).exists() else None for k in ['started_at.txt','finished_at.txt','exit_code.txt','wrapper.pid','launch_attempt.json']}
log=(rt/'driver.log').read_text(errors='replace') if (rt/'driver.log').is_file() else ''
d={'time':datetime.datetime.now(tz).isoformat(),'runtime':str(rt),'state':state,'wrapper_alive':bool(state['wrapper.pid']) and Path('/proc',state['wrapper.pid']).exists(),'completed_steps':re.findall(r'Global Step:\s*(\d+)\s*/',log)[-3:],'resume_evidence':[l for l in log.splitlines() if any(x in l.lower() for x in ['resum','load checkpoint','loading checkpoint','local shard','local_shard','start training'])][-50:],'driver_tail':log.splitlines()[-30:],'errors':{k:log.count(k) for k in ['Fatal Python error','CUDA out of memory','RuntimeError:','Traceback (most recent call last):']}}
ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
d['scalars']={k:[{'step':x.step+1,'value':x.value,'wall_time':x.wall_time} for x in ea.Scalars(k)][-10:] for k in ['env/success_once','eval/success_once','time/step']}
d['gpu45']=subprocess.run(['nvidia-smi','-i','4,5','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
d['checkpoint_steps']=sorted([int(p.name.split('_')[-1]) for p in run.glob('*/checkpoints/global_step_*') if p.is_dir()])
d['checkpoint200_files']=[{'path':str(p),'bytes':p.stat().st_size} for p in run.glob('*/checkpoints/global_step_200/**/*') if p.is_file()]
d['disk']=subprocess.run(['df','-B1','/data'],capture_output=True,text=True).stdout
print(json.dumps(d))
PY
