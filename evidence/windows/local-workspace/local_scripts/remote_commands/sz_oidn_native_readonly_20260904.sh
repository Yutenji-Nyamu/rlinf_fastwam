#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import re,subprocess,json
root=Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages/sapien')
for p in (root/'_oidn_tricks.py',root/'wrapper/scene.py'):
 lines=p.read_text().splitlines();sel=set(range(len(lines))) if '_oidn' in p.name else set()
 for i,l in enumerate(lines):
  if re.search(r'def (__del__|clear)|RenderSystem|render_system',l):sel.update(range(max(0,i-3),min(len(lines),i+30)))
 print('SOURCE',p);print('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(sel)))
print('LIBC',subprocess.run(['getconf','GNU_LIBC_VERSION'],capture_output=True,text=True).stdout)
for cg in ('/sys/fs/cgroup/user.slice/user-1003.slice','/sys/fs/cgroup/user.slice/user-1003.slice/session-1419.scope'):
 for key in ('memory.max','memory.current','memory.events','pids.max','pids.current','pids.events'):
  p=Path(cg)/key
  if p.exists():print(str(p),p.read_text().strip())
for repo in ('fastwam-current-grpo','sidney-pi05-current-rlinf','robotwin-vector-render-lifecycle-fix-0008ae6'):
 p=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')/repo
 print('GIT',repo)
 for args in (['rev-parse','HEAD'],['status','--porcelain']):print(subprocess.run(['git','--no-optional-locks','-C',str(p),*args],capture_output=True,text=True).stdout)
print('SOURCE_FILE_PATHS')
wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/fastwam-current-grpo/rlinf/workers/env/env_worker.py')
lines=wt.read_text().splitlines()
for lo,hi in ((865,925),(1040,1095)):
 print('\n'.join(f'{i+1}: {lines[i]}' for i in range(lo-1,hi)))
PY
