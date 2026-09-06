#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
date -Is
id
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
free -b
df -h /data /home
ps -u chenyiteng -o pid,pgid,etime,rss,args | grep -E 'gcs_server|raylet|train_embodied|EnvWorker|RolloutWorker' | cut -c1-430
python3 - <<'PY'
from pathlib import Path
import re,subprocess
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
runs=[base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1',base/'fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2']
for run in runs:
 print('RUN',run)
 for n in ['wrapper.pid','exit_code.txt','finished_at.txt','started_at.txt']:
  p=run/'runtime'/n
  print(n,p.read_text().strip() if p.exists() else 'absent')
 log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(run/'runtime/driver.log').read_text(errors='replace'))
 print('LAST_STEPS',[l[-1500:] for l in log.splitlines() if 'Global Step:' in l][-2:])
 for key in ['pthread_key_create failed','Fatal Python error','CUDA out of memory','Traceback']:
  print('ERROR',key,log.count(key))
 for p in sorted(run.glob('**/global_step_30/**')):
  if p.is_file() and (p.name=='.metadata' or p.suffix=='.distcp'): print('CHECKPOINT',p.relative_to(run),p.stat().st_size)
 if 'fastwam' in str(run):
  for n in ['command.txt','contract.json']: print(n,(run/'runtime'/n).read_text())
root=Path('/data/chenyiteng/projects/rlinf-shenzhen')
for repo in [root/'worktrees/fastwam-current-grpo',root/'RoboTwin-RLinf-support',root/'worktrees/robotwin-vector-render-lifecycle-fix-0008ae6',root/'worktrees/robotwin-oidn-toggle-20260904']:
 print('REPO',repo)
 for args in [['rev-parse','HEAD'],['status','--porcelain'],['remote','-v']]:print(subprocess.run(['git','--no-optional-locks','-C',str(repo),*args],capture_output=True,text=True,check=True).stdout)
print('NEW_TARGET_EXISTS',(root/'worktrees/robotwin-clean-oidn-off-20260904').exists())
rl=root/'worktrees/fastwam-current-grpo'
for path in [rl/'examples/embodiment/train_embodied_agent.py',rl/'rlinf/scheduler/cluster/cluster.py']:
 if path.exists():
  print('RUNTIME_SOURCE',path)
  lines=path.read_text().splitlines()
  hits={j for i,l in enumerate(lines) if any(x in l for x in ['namespace','RLINF_CODE_WORKING_DIR','PYTHONPATH','runtime_env']) for j in range(max(0,i-4),min(len(lines),i+12))}
  for i in sorted(hits):print(f'{i+1}: {lines[i]}')
PY
RAY_ADDRESS=172.17.0.1:6389 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/ray status
