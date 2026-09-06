#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
date -Is
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
free -b
ps -p 321933,322685,3176215 -o user,pid,etime,stat,comm
python3 - <<'PY'
from pathlib import Path
import os,re,subprocess,json
run=Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1')
print('RUN',run)
for name in ['wrapper.pid','observer.pid','started_at.txt','exit_code.txt','finished_at.txt']:
 p=run/'runtime'/name;print(name,p.read_text().strip() if p.exists() else 'absent')
 if p.exists() and name.endswith('.pid'):
  pid=int(p.read_text()); print(subprocess.run(['ps','-p',str(pid),'-o','user,pid,pgid,etime,stat,args'],capture_output=True,text=True).stdout)
log=(run/'runtime/driver.log').read_text(errors='replace')
log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',log)
for key in ['OIDN Error','pthread_key_create failed','Fatal Python error','CUDA out of memory','Traceback','RayActorError']:
 print('ERROR',key,log.count(key))
for key in ['Global Step:','Rollout Epoch:','ray_tracing_denoiser','resume_dir','namespace']:
 lines=[l for l in log.splitlines() if key in l];print('MATCH',key,'\n'.join(lines[-8:]))
print('TAIL','\n'.join(log.splitlines()[-35:]))
p=run/'runtime/resource.csv'
if p.exists():print('RESOURCE','\n'.join(p.read_text().splitlines()[-4:]))
for p in Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=os.getuid():continue
  cmd=(p/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')
  if not ('ray::EnvWorker' in cmd or 'ray::MultiStepRolloutWorker' in cmd or ('train_embodied_agent.py' in cmd and str(run) in cmd)):continue
  env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
  if env.get('ROBOTWIN_PATH','').endswith('robotwin-clean-oidn-off-20260904'):
   print('OWNED_BINDING',p.name,cmd[:200],json.dumps({k:env.get(k) for k in ['ROBOTWIN_PATH','PYTHONPATH','CUDA_VISIBLE_DEVICES','RLINF_CODE_WORKING_DIR']},ensure_ascii=False))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
sid=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1/runtime/driver.log')
s=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',sid.read_text(errors='replace'))
print('SIDNEY_LAST_STEP',[l for l in s.splitlines() if 'Global Step:' in l][-1:])
print('SIDNEY_ERRORS',{k:s.count(k) for k in ['OIDN Error','Fatal Python error','CUDA out of memory','Traceback']})
PY
