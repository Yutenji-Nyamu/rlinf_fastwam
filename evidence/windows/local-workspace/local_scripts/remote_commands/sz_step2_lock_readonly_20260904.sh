#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
date -Is
SP=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/py-spy
timeout 15s "$SP" dump --pid 1053120 --native --nonblocking
python3 - <<'PY'
from pathlib import Path
import json,subprocess
cmd=['/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/py-spy','dump','--pid','1053120','--nonblocking','--json','-ll']
r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
print('LOCALS_STATUS',r.returncode,r.stderr)
try:
 data=json.loads(r.stdout)
 print('JSON_KIND',type(data).__name__)
 if isinstance(data,list):
  for thread in data:
   chosen=[]
   for frame in thread.get('frames',[]):
    if frame.get('name') in ['_get_module_lock','acquire','__enter__','_find_spec','available_solvers']:
     chosen.append(frame)
   if chosen:print('IMPORT_THREAD',json.dumps({k:v for k,v in thread.items() if k!='frames'},default=str),json.dumps(chosen,default=str))
except Exception as e: print('JSON_PARSE',repr(e),r.stdout[:500])
root=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
r=subprocess.run(['git','-C',str(root/'robotwin-vector-render-lifecycle-fix-0008ae6'),'diff','0008ae6800df9f75fc8de7098bacb01735fd8fd2','HEAD','--','envs/vector_env.py'],capture_output=True,text=True)
print('OLD_PATCH',r.stdout,r.stderr)
for tid in [1065920,1066014,1066016,1066019,1066195,1066198,1066199]:
 p=Path('/proc/1053120/task')/str(tid)
 print('THREAD_STATE',tid)
 for n in ['wchan','syscall','stack']:
  try:print(n,(p/n).read_text().strip())
  except OSError as e:print(n,e)
PY
date -Is
