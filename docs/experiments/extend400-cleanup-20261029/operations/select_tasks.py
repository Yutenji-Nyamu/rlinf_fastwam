import json
from pathlib import Path
W=Path('C:/Users/86136/Documents/rl/worktrees/pi05-distill-20261021')
D=Path(__file__).resolve().parent
B=W/'docs/experiments/c10-distill-restart-20261025/evidence/closed-sweep'
limits={x['task']:x['step_limit'] for x in json.loads((W/'docs/experiments/onedp-closeout-chunk-sweep-20261024/evidence/sweep/task-list.json').read_text())['tasks']}
rows={}
for n in [1,10]:
 for b in [0,1]:
  f=B/f'summary-raw-c10-n{n}-b{b}.json'
  for r in json.loads(f.read_text())['tasks']:
   if not r['valid']:continue
   x=rows.setdefault(r['task'],dict(task=r['task'],step_limit=limits[r['task']],n1=[0,0],n10=[0,0]))
   x[f'n{n}'][0]+=r['successes'];x[f'n{n}'][1]+=r['episodes']
short=[x for x in rows.values() if x['step_limit']==400 and x['n1'][1] and x['n10'][1]]
# Figure-oriented preliminary task pool: short tasks with usable student successes.
short.sort(key=lambda x:(-min(x['n1'][0]/x['n1'][1],x['n10'][0]/x['n10'][1]),-x['n10'][0]/x['n10'][1],x['task']))
report={'status':'provisional pending student/teacher pair; historical 16-episode rates select tasks, never seeds','history':'RoboTwin Clean H50/C10 original Sidney pi05; not native single-task teacher','candidates':short}
(D/'signal-task-candidates.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
for x in short:print(x)
