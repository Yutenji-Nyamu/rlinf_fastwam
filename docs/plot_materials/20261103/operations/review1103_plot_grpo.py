import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
p=Path('/data/chenyiteng/exports/plot-backup-20261103');d=json.loads((p/'grpo-series.json').read_text())
labels=['C15 Clean 256 / B1024','C16 Clean 128 / B512','C17 Clean 128 / B512','U tau1','U tau2','U tau3','Signed U tau1','Signed U tau2','Signed U tau3 interrupted','Signed U tau3 resume-A','Signed U tau3 resume-B','Clean current']
fig,ax=plt.subplots(figsize=(13,5.8),layout='constrained')
colors=['#0072B2','#56B4E9','#009E73','#CC79A7','#E69F00','#D55E00','#6a3d9a','#17becf','#a6761d','#999999','#666666','#111111']
for i,(name,rows)in enumerate(d.items()):
 x,y=zip(*rows);ax.plot(x,y,label=labels[i],color=colors[i],alpha=1 if i==11 else .65,lw=2.8 if i==11 else 1.1,marker='o'if i==11 else None,markersize=4)
ax.set(xlabel='Training round',ylabel='Evaluation success (%)',xlim=(0,200),ylim=(0,100),title='Pillbottle GRPO family | current Clean versus historical runs')
ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8,loc='upper left',bbox_to_anchor=(0,-.16))
for ext in ['png','svg','pdf']:fig.savefig(p/f'grpo-family.{ext}',dpi=170)
print('GRPO family figure saved; historical input snapshot 2026-10-24, current input 2026-11-03')

d=json.loads((p/'single-points.json').read_text());fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
names=[('Direct',11),('CP v2',30),('DMD new',13),('MFD',13),('Direct + GAD',11),('CP v2 + GAD',30),('DMD new + GAD',13),('MFD + GAD',13),('Direct + GAD-OT',11),('FQL v2',11)]
cols=['#0072B2','#009E73','#D55E00','#7B2CBF']
for i,(name,scale)in enumerate(names):
 rows=sorted([x for x in d if x['method']==name and x['protocol']=='Native single task'],key=lambda x:x['step'])
 if not rows:continue
 color=cols[i%4]if i<8 else ['#C51B7D','#222222'][i-8]
 ax.plot([x['step']/scale for x in rows],[x['rate']for x in rows],label=name,color=color,ls='--'if 4<=i<8 else '-',marker=['o','s','^','D'][i%4],markersize=3,lw=1.6)
ax.set(xlim=(0,400),ylim=(0,100),xlabel='Training budget epoch',ylabel='Success (%)',title='Single-task adjust_bottle | 128 episodes per evaluation')
ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=9,loc='upper left',bbox_to_anchor=(0,-.16))
for ext in ['png','svg','pdf']:fig.savefig(p/f'single-family.{ext}',dpi=170)
