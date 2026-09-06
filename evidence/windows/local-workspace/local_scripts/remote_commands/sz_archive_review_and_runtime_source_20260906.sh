set -eu
export PYTHONDONTWRITEBYTECODE=1 GIT_TERMINAL_PROMPT=0
nice -n 19 ionice -c 3 /usr/bin/python3 -B - <<'PY'
import json,subprocess
from pathlib import Path
base=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc');tree=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
res={}
for name,pattern in [('pi05-online-bc','pi05-pillbottle-bc32x1*formal100*'),('pi05-online-bc-dvac','pi05-pillbottle-bc-dvac32x1*formal100*')]:
 run=list(base.glob(pattern));assert len(run)==1
 start=(run[0]/'runtime/source-head.txt').read_text().strip()
 diff=subprocess.run(['git','-C',str(tree/name),'diff','--name-status',start,'HEAD','--','rlinf','examples','tests'],capture_output=True,text=True,check=True)
 res[name]={'startup_head':start,'runtime_relevant_changes_since_start':diff.stdout}
print('RUNTIME_SOURCE_RECHECK='+json.dumps(res),flush=True)
archive=tree/'experiment-archive-20260906'
assert subprocess.check_output(['git','-C',str(archive),'branch','--show-current'],text=True).strip()=='codex/sz-experiment-archive-20260906'
print('AUTHOR='+subprocess.check_output(['git','-C',str(archive),'var','GIT_AUTHOR_IDENT'],text=True).strip(),flush=True)
print('ARCHIVE_READY_FILES='+str(sum(1 for p in (archive/'evidence').rglob('*') if p.is_file())),flush=True)
PY
