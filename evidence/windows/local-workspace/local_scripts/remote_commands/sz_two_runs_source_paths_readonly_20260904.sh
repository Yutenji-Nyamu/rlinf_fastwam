#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import subprocess,re,datetime,json
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
runs=[base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1',base/'fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2']
print('UTC',datetime.datetime.now(datetime.timezone.utc).isoformat())
for r in runs:
 print('RUN',r.name)
 print('COMMAND', (r/'runtime/command.txt').read_text())
 print('CONTRACT',(r/'runtime/contract.json').read_text())
 log=(r/'runtime/driver.log').read_text(errors='replace')
 print('PROGRESS', re.findall(r'Global Step:\s*\d+/100',log)[-1:])
 print('LAST_ROLLOUT', [l.strip() for l in log.splitlines() if 'Generating Rollout Epochs:' in l][-1:])
 print('OOM_AND_NONFINITE',len(re.findall(r'CUDA out of memory|OutOfMemoryError|non.?finite',log,re.I)))
print('FASTWAM_POST_LAUNCH_DIFF')
p=subprocess.run(['git','--no-optional-locks','-C','/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo','diff','--stat','7b2331c55d14397cfb4cb16181470ddc8afae44a','HEAD'],capture_output=True,text=True)
print(p.stdout,p.stderr)
print('ROBOTWIN_WORKTREES')
p=subprocess.run(['git','--no-optional-locks','-C','/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support','worktree','list','--porcelain'],capture_output=True,text=True)
print(p.stdout,p.stderr)
PY
