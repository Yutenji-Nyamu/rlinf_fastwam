#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,json,re,subprocess,os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
tz=datetime.timezone(datetime.timedelta(hours=8))
root=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
ea=EventAccumulator(str(root/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
scalars={k:[{'step':x.step+1,'value':x.value,'wall_time':x.wall_time} for x in ea.Scalars(k)] for k in ['env/success_once','eval/success_once','time/step']}
log=(root/'runtime/driver.log').read_text(errors='replace')
state={k:(root/'runtime'/k).read_text().strip() if (root/'runtime'/k).exists() else None for k in ['started_at.txt','finished_at.txt','exit_code.txt','wrapper.pid','driver.pid']}
d={'time':datetime.datetime.now(tz).isoformat(),'identity':{'user':subprocess.check_output(['id','-un'],text=True).strip(),'uid':os.getuid()},'run':str(root),'scalars':scalars,'completed_steps':re.findall(r'Global Step:\s*(\d+)\s*/',log)[-1:],'state':state,'wrapper_alive':bool(state['wrapper.pid']) and Path('/proc',state['wrapper.pid']).exists(),'phase':[l for l in log.splitlines() if 'Generating Rollout Epochs:' in l][-3:],'errors':{k:log.count(k) for k in ['Fatal Python error','CUDA out of memory','RuntimeError:','Traceback (most recent call last):']},'driver_tail':log.splitlines()[-24:]}
d['processes']={k:subprocess.run(['ps','-p',v,'-o','pid,ppid,stat,etime,args'],capture_output=True,text=True).stdout for k,v in state.items() if k.endswith('.pid') and v and v.isdigit()}
d['gpu45']=subprocess.run(['nvidia-smi','-i','4,5','--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
d['disk']=subprocess.run(['df','-B1','/data'],capture_output=True,text=True).stdout
d['checkpoint_files']={str(step):[{'path':str(p),'bytes':p.stat().st_size,'mtime':datetime.datetime.fromtimestamp(p.stat().st_mtime,tz).isoformat()} for p in root.glob('*/checkpoints/global_step_'+str(step)+'/**/*') if p.is_file()] for step in [90,100]}
print(json.dumps(d))
PY
