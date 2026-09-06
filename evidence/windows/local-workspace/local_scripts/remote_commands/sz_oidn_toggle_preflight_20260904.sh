#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1
date -Is
id
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
free -b
df -h /data /home
ps -u chenyiteng -o pid,pgid,etime,rss,args | grep -E 'gcs_server|raylet|train_embodied|EnvWorker|RolloutWorker' | cut -c1-650
python3 - <<'PY'
from pathlib import Path
import re,subprocess,json,datetime
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
runs=[base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1',base/'fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2']
for run in runs:
 print('RUN',run)
 for n in ['wrapper.pid','exit_code.txt','finished_at.txt']:
  p=run/'runtime'/n
  print(n,p.read_text().strip() if p.exists() else 'absent')
 log=(run/'runtime/driver.log').read_text(errors='replace')
 log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',log)
 print('LAST_STEPS',[l[-1400:] for l in log.splitlines() if 'Global Step:' in l][-2:])
 print('LAST_EVAL',[l[-1400:] for l in log.splitlines() if 'eval/' in l][-2:])
 for key in ['pthread_key_create failed','Fatal Python error','CUDA out of memory','Traceback']:
  print('ERROR',key,log.count(key))
 for p in sorted(run.glob('**/global_step_30/**')):
  if p.is_file(): print('CHECKPOINT',p.relative_to(run),p.stat().st_size)
 if 'fastwam' in str(run):
  for n in ['command.txt','contract.json']:
   p=run/'runtime'/n
   if p.exists():print(n,p.read_text())
rl=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo')
rt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6')
for repo in [rl,rt]:
 print('REPO',repo)
 for args in [['rev-parse','HEAD'],['status','--porcelain']]:print(subprocess.run(['git','--no-optional-locks','-C',str(repo),*args],capture_output=True,text=True).stdout)
for p,lo,hi in [(rt/'envs/_base_task.py',170,240),(rl/'rlinf/envs/robotwin/robotwin_env.py',1,180)]:
 print('SOURCE',p)
 print('\n'.join(f'{i+1}: {l}' for i,l in enumerate(p.read_text().splitlines()) if lo<=i+1<=hi))
print('INFERENCE_CANDIDATES')
print(subprocess.run(['rg','--files',str(rl/'tests'),str(rl/'examples'),str(rl/'scripts')],capture_output=True,text=True).stdout[:30000])
PY
