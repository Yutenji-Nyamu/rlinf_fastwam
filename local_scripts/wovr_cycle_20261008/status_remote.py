import json, subprocess, time
from pathlib import Path
root=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/wm-cycle-20261009-v2')
out=root/'run-retry2'
def read(p): return json.loads(p.read_text()) if p.exists() else None
def tail(p,n=8):
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-16000)); text=f.read().decode('utf-8',errors='replace')
  print('TAIL',str(p),'\n'.join(x[:700] for x in text.splitlines()[-n:]))
print('TIME',time.strftime('%F %T'))
for name in ['owner.json','status.json','error.json','final.json']:
 value=read(out/name)
 if value is not None: print(name,json.dumps(value))
owner=read(out/'owner.json') or {}
p=Path('/proc')/str(owner.get('pid',0))/'stat'
if p.exists():
 stat=p.read_text().split(') ',1)[1].split();print('OWNER_LIVE',int(stat[19])==owner.get('start'),stat[0])
phase=(read(out/'status.json') or {}).get('phase','none')
tail(out/phase/'process.log',12)
for p in sorted(out.glob('rm*/train/report.json')): print('RM_REPORT',p.read_text())
for p in sorted(out.glob('offline_*.json')): print('OFFLINE',p.name,p.read_text())
for p in sorted(out.glob('rl*/worker_metrics/*')):
 if p.is_file():tail(p,3)
queue=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
for name in ['owner-identity.json','status.json']:
 value=read(queue/name)
 if value is not None:print('QUEUE',name,json.dumps(value))
print('GPU',subprocess.run(['nvidia-smi','-i','4,5','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=15).stdout)
for pid,start in [(1320019,743498084),(1320020,743498089)]:
 p=Path('/proc')/str(pid)/'stat'
 print('BC67',pid,p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==start)
