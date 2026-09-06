#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
date -Is
for candidate in /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/py-spy /usr/local/bin/py-spy /home/chenyiteng/.local/bin/py-spy; do
  if test -x "$candidate"; then
    printf 'PYSPY=%s\n' "$candidate"
    for pid in 1053120 1053117 1053114 1053118; do
      timeout 10s "$candidate" dump --pid "$pid" --nonblocking 2>&1
    done
    break
  fi
done
python3 - <<'PY'
import time,json,subprocess
from pathlib import Path
pids=[1052633,1053112,1053114,1053115,1053117,1053118,1053120]
def snap():
 out=[]
 for pid in pids:
  p=Path('/proc')/str(pid)
  try:
   stat=(p/'stat').read_text().split(') ',1)[1].split()
   out.append({'pid':pid,'cpu_ticks':int(stat[11])+int(stat[12]),'state':stat[0],'wchan':(p/'wchan').read_text().strip(),'cmd':(p/'cmdline').read_bytes().decode(errors='replace').strip('\0 ').replace('\0',' ')[:120]})
  except OSError as e:out.append({'pid':pid,'error':str(e)})
 return out
print('FIRST',json.dumps(snap()))
time.sleep(5)
print('SECOND',json.dumps(snap()))
root=Path('/data/chenyiteng/ray/rlt-dsrl-v3/session_2026-08-23_16-27-53_911161_321906/logs')
for pid in [1053120,1053117]:
 for p in sorted(root.glob(f'worker-*-{pid}.*')):
  print('WORKER_LOG',p,'mtime',p.stat().st_mtime)
  print('\n'.join(p.read_text(errors='replace').splitlines()[-20:]))
PY
nvidia-smi -i 6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
date -Is
