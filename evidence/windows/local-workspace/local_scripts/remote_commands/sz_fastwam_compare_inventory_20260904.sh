#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import datetime,subprocess
print('CST',datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
for cmd in (['id'],['uptime'],['free','-b'],['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],['ps','-u','chenyiteng','-o','pid,ppid,etime,stat,rss,comm']):
 print('COMMAND',cmd);r=subprocess.run(cmd,capture_output=True,text=True,timeout=20);print(r.stdout)
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
for p in sorted(base.iterdir()):
 if p.is_dir():
  print('RESULT_GROUP',p)
  for runs in [p/'runs']:
   if runs.exists():
    for run in sorted(runs.iterdir()):
     if run.is_dir():
      print('RUN',run.name)
      for name in ('exit_code.txt','finished_at.txt'):
       f=run/'runtime'/name
       if f.exists():print(name,f.read_text().strip())
      for pat in ('runtime/*resolved*','*events*','tensorboard/*','*/checkpoints/global_step_*'):
       files=list(run.glob(pat));print(pat,[(str(f.relative_to(run)),f.stat().st_size) for f in files][-12:])
for p in sorted(Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees').iterdir()):
 if p.is_dir():
  print('WORKTREE',p)
  if any(s in p.name for s in ('fastwam-current','sidney-pi05','robotwin-vector')):
   for args in (['rev-parse','HEAD'],['status','--short']):print(subprocess.run(['git','--no-optional-locks','-C',str(p),*args],capture_output=True,text=True).stdout)
PY
