#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import datetime,subprocess,hashlib,re,json
print('CST',datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
print(subprocess.run(['id'],capture_output=True,text=True).stdout)
rt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6')
rl=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo')
sp=Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/sapien')
for p in (rt,rl):
 print('GIT',p)
 for args in (['rev-parse','HEAD'],['status','--porcelain']): print(subprocess.run(['git','--no-optional-locks','-C',str(p),*args],capture_output=True,text=True).stdout)
files=[rt/'robotwin/envs/vector_env.py',rt/'envs/_base_task.py',rt/'envs/move_stapler_pad.py',sp/'wrapper/scene.py',sp/'wrapper/engine.py',sp/'_oidn_tricks.py',rl/'rlinf/envs/robotwin/robotwin_env.py',rl/'rlinf/workers/env/env_worker.py']
for p in files:
 if not p.exists():print('MISSING',p);continue
 lines=p.read_text().splitlines();print('SOURCE',p,'SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
 if p.name in ('vector_env.py','move_stapler_pad.py','engine.py','_oidn_tricks.py'): selected=set(range(len(lines)))
 else:
  ranges={'_base_task.py':[(1,240),(380,480),(610,680)],'scene.py':[(1,115),(400,440)],'robotwin_env.py':[(1,170),(205,275),(420,590)],'env_worker.py':[(115,205),(850,930),(950,1090),(1330,1455)]}[p.name]
  selected={i for lo,hi in ranges for i in range(lo-1,min(hi,len(lines)))}
 print('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(selected)))
print('NATIVE_DEPENDENCIES')
for p in sp.glob('*.so'):
 print('LIBRARY',p);print(subprocess.run(['ldd',str(p)],capture_output=True,text=True).stdout)
print('NATIVE_LOADER_PATHS_IN_OWN_WORKER')
for pid in (3177205,3177207):
 p=Path('/proc')/str(pid)/'maps'
 if p.exists():
  lines=p.read_text().splitlines();print('PID',pid)
  print('\n'.join(sorted({l.split()[-1] for l in lines if any(x in l.lower() for x in ('oidn','imagedenoise','libtbb','libc.so','sapien'))})))
root=Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2')
for name in ('exit_code.txt','finished_at.txt'):print('FAST_TERMINAL',name,(root/'runtime'/name).read_text().strip())
log=(root/'runtime/driver.log').read_text(errors='replace')
for key in ('pthread_key_create failed','OIDN Error: invalid handle','Fatal Python error'):
 hits=[(i+1,l[:400]) for i,l in enumerate(log.splitlines()) if key in l];print('FATAL_REFRESH',key,len(hits),hits[:2])
for p in root.rglob('*resolved*.yaml'):
 print('RESOLVED',p)
 lines=p.read_text().splitlines()
 for i,l in enumerate(lines):
  if re.search(r'offload|clear_cache|auto_reset|total_num_envs|denois',l):print(i+1,l)
PY
