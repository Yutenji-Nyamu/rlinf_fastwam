#!/usr/bin/env bash
set -eu
id
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,json,os,re,subprocess
from pathlib import Path
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
assert os.getuid()==1003
def command(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    return {'rc':p.returncode,'out':p.stdout,'err':p.stderr}
def read(path):
    return path.read_text(errors='replace') if path.is_file() else ''
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
pi05=base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1'
old=base/'grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2'
bc=base/'online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v6'
fast=base/'fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3'
def inspect(run,rt):
    log=read(rt/'driver.log')
    if not log: log=read(run/'driver.log')
    states={k:read(rt/k).strip() for k in ['started_at.txt','finished_at.txt','exit_code.txt','wrapper.pid']}
    pid=states['wrapper.pid']
    d={'run':str(run),'state':states,'wrapper_alive':bool(pid) and Path('/proc',pid).exists(),
       'completed_steps':re.findall(r'Global Step:\s*(\d+)\s*/',log)[-5:],
       'tail':log.splitlines()[-10:],
       'errors':{s:log.count(s) for s in ['Fatal Python error','CUDA out of memory','RuntimeError:','Traceback (most recent call last):','getSemaphoreFdKHR','cannot create buffer','Too many open files']}}
    tb=run/'tensorboard'
    if tb.is_dir():
        ea=EventAccumulator(str(tb),size_guidance={'scalars':0});ea.Reload()
        d['scalars']={k:[{'step':x.step+1,'value':x.value} for x in ea.Scalars(k)][-12:] for k in ['env/success_once','eval/success_once','time/step'] if k in ea.Tags()['scalars']}
    dirs=sorted(set(p for p in run.glob('**/checkpoints/global_step_*') if p.is_dir()),key=lambda p:int(p.name.split('_')[-1]))
    d['checkpoint_steps']=[int(p.name.split('_')[-1]) for p in dirs]
    d['latest_checkpoint_files']=[{'path':str(f.relative_to(run)),'bytes':f.stat().st_size} for p in dirs[-1:] for f in p.rglob('*') if f.is_file()]
    return d
d={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
   'pi05':inspect(pi05,pi05/'runtime-resume100-to200'),'bc':inspect(bc,bc),'fast':inspect(fast,fast/'runtime'),'old_control':inspect(old,old/'runtime')}
cfg=yaml.safe_load(read(old/'runtime/resolved.yaml'))
d['old_config']={s:cfg.get(s) for s in ['algorithm','actor','env','rollout','runner']}
tree='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc'
d['bc_git']={s:command(['git','--no-optional-locks','-C',tree,*args]) for s,args in {'head':['rev-parse','HEAD'],'status':['status','--porcelain']}.items()}
for name,args in {'gpu':['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],
                  'memory':['free','-b'],'disk':['df','-B1','/','/home','/data'],
                  'vmstat':['vmstat','1','2'],'load':['uptime'],
                  'own_workers':['ps','-u','chenyiteng','-o','pid,ppid,etime,pcpu,rss,comm']}.items():
    d[name]=command(args)
d['pressure']={k:read(Path('/proc/pressure')/k) for k in ['cpu','memory','io']}
d['ray_limits']={str(pid):read(Path('/proc',str(pid),'limits')) for pid in [321933,322685]}
print('FOLLOWUP_JSON '+json.dumps(d,ensure_ascii=False))
PY
