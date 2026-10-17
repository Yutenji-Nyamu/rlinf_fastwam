import os,sys,json,time,subprocess
from pathlib import Path
T=Path('/data/chenyiteng/deployment-20261017/pi05-policy-budget-eval-v1');O=Path('/data/chenyiteng/results/pi05-policy-eval/pillbottle-20261017-v1');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0,str(S/'ops'));from common import read,same,gpus
out=dict(time=time.time(),gpus=gpus(),lanes={},owner=read(S/'owner-identity.json'),state=read(S/'status.json'))
out['owner_alive']=same(out['owner']);out['memory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
for g in (6,7):
 d={}
 for name in [f'lane-g{g}-status.json',f'lane-g{g}-error.json',f'lane-g{g}-released.json']:
  if (T/name).exists():d[name]=read(T/name)
 status=d.get(f'lane-g{g}-status.json',{})
 if 'config' in status:
  c=read(status['config']);b=Path(c['output'])
  for name in ['progress.json','started.json','done.json','strong-loaded.json','error.json']:
   if (b/name).exists():d[name]=read(b/name)
  log=T/(b.name+f'-g{g}.log')
  if log.exists():
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2200));d['log_tail']=f.read().decode(errors='replace')
 out['lanes'][g]=d
out['summaries']=read(O/'summary.json') if (O/'summary.json').exists() else []
print(json.dumps(out))
