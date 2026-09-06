#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import os, re, json, datetime, subprocess, statistics
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml
tz=datetime.timezone(datetime.timedelta(hours=8))
def stamp(t): return datetime.datetime.fromtimestamp(t,tz).isoformat()
runs = {
 'sidney': Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1'),
 'fastwam': Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2'),
}
print('AUDIT_CST',stamp(datetime.datetime.now().timestamp()),flush=True)
for label, run in runs.items():
 print('RUN',label,flush=True)
 log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'runtime/driver.log').read_text(errors='replace'))
 print('LAST_GLOBAL_STEP',re.findall(r'Global Step:\s*(\d+)/100',log)[-1:])
 print('LAST_PHASE_LINES',[line.strip()[:450] for line in log.splitlines() if re.search(r'Generating Rollout Epochs:|Evaluating|Evaluation|Saving checkpoint|Evaluating Epochs',line)][-12:])
 lines=log.splitlines()
 fail=[(i,line) for i,line in enumerate(lines) if re.search(r'OIDN.*Error|pthread_key_create|Fatal Python error|Traceback|CUDA out of memory|OutOfMemoryError',line)]
 print('FATAL_FIRST',[(i+1,line[:550]) for i,line in fail[:5]])
 print('FATAL_KIND_COUNTS',{s:sum(bool(re.search(s,l)) for l in lines) for s in ['OIDN.*Error','pthread_key_create','Fatal Python error','Traceback','CUDA out of memory','OutOfMemoryError']})
 if fail:
  start=fail[0][0]
  print('BEFORE_FIRST_FATAL',[l[:450] for l in lines[max(0,start-16):start]])
  print('FATAL_PYTHON_LINES',[l[:700] for l in lines if 'Fatal Python error' in l])
 for name in ('started_at.txt','finished_at.txt','exit_code.txt','launch_manifest.txt','source_head.txt','source-head.txt'):
  p=run/'runtime'/name
  if p.exists(): print(name,p.read_text().strip())
 cfg=yaml.safe_load((run/'runtime/resolved.yaml').read_text())
 print('RESOLVED_PATHS',json.dumps({'runner':cfg.get('runner'),'assets_path':cfg.get('env',{}).get('train',{}).get('assets_path')},ensure_ascii=False))
 print('ROOT_DIRS',[str(p.relative_to(run)) for p in run.iterdir() if p.is_dir()])
 print('METRICS_HEAD',(run/'metrics.log').read_text().splitlines()[0][:1200])
 print('METRICS_TAIL',(run/'metrics.log').read_text().splitlines()[-1][:1600])
 ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0}); ea.Reload()
 print('SCALAR_TAGS',ea.Tags().get('scalars',[]))
 for tag in ea.Tags().get('scalars',[]):
  if re.search(r'success|train_time|step_time|rollout_time|elapsed|actor/grad_norm|actor/approx_kl',tag):
   points=ea.Scalars(tag)
   rows=[{'step':p.step,'value':p.value,'cst':stamp(p.wall_time)} for p in points]
   out={'tag':tag,'count':len(rows),'first':rows[:2],'last':rows[-12:]}
   if 'success' in tag:
    out['all']=rows
    out['MA5']=statistics.mean([p.value for p in points[-5:]])
    out['MA10']=statistics.mean([p.value for p in points[-10:]])
   print('SCALAR',json.dumps(out))
 checkpoints=[]
 for root,dirs,files in os.walk(run):
  dirs[:]=[d for d in dirs if d not in ('robotwin_data','tensorboard','runtime','videos','video','wandb')]
  for name in files:
   if name.endswith(('.pt','.pth','.distcp')) or name in ('.metadata','metadata.json'):
    p=Path(root)/name; st=p.stat()
    checkpoints.append({'path':str(p.relative_to(run)),'bytes':st.st_size,'cst':stamp(st.st_mtime)})
 print('CHECKPOINT_FILES',json.dumps(checkpoints))
 print('LOG_MTIME',stamp((run/'runtime/driver.log').stat().st_mtime))
 print('LOG_LAST_PROGRESS',[l.strip()[:350] for l in log.splitlines() if 'Generating Rollout Epochs:' in l][-2:])
for repo in ('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf','/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo','/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support'):
 print('LIVE_GIT',repo)
 for args in (['rev-parse','HEAD'],['branch','--show-current'],['status','--porcelain','--untracked-files=normal']):
  p=subprocess.run(['git','--no-optional-locks','-C',repo,*args],capture_output=True,text=True)
  print(json.dumps({'args':args,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
print('OWNED_ACTIVE_CORE')
p=subprocess.run(['ps','-u',str(os.getuid()),'-o','pid,ppid,pgid,etime,stat,%cpu,rss,args'],capture_output=True,text=True)
for l in p.stdout.splitlines():
 if ('train_embodied_agent.py' in l or re.search(r'ray::(?:Embodied|EnvWorker|MultiStep)|/gcs_server|/raylet ',l)) and 'python -' not in l:
  print(l[:500])
print('GPU_FINAL')
print(subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout)
print('RAM_PSI_FINAL')
print(subprocess.run(['free','-b'],capture_output=True,text=True).stdout)
print(Path('/proc/pressure/memory').read_text(),Path('/proc/pressure/io').read_text())
PY
