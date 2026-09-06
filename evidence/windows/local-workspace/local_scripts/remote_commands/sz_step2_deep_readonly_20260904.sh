#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
date -Is
id
SP=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/py-spy
"$SP" dump --help
timeout 12s "$SP" dump --pid 1053120 --nonblocking
python3 - <<'PY'
from pathlib import Path
root=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-clean-oidn-off-20260904')
site=Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages')
specs=[(site/'toppra/solverwrapper/solverwrapper.py',1,110),(site/'toppra/algorithm/reachabilitybased/reachability_algorithm.py',30,135),(site/'mplib/planner.py',325,380),(root/'envs/vector_env.py',1,430),(root/'envs/_base_task.py',1800,1945),(root/'envs/camera/camera.py',305,360),(root/'envs/move_stapler_pad.py',105,140)]
for p,lo,hi in specs:
 print('\nSOURCE',p,lo,hi)
 try:
  lines=p.read_text().splitlines()
  for i in range(lo-1,min(hi,len(lines))): print(f'{i+1:5}: {lines[i]}')
 except OSError as e:print(e)
p=Path('/proc/1053120/task')
for t in sorted(p.iterdir(),key=lambda v:int(v.name)):
 if int(t.name)<1065900:continue
 print('THREAD',t.name)
 for f in ['comm','wchan','syscall']:
  try:print(f,(t/f).read_text().strip())
  except OSError as e:print(f,e)
PY
date -Is
