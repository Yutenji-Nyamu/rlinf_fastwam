"""Render the complete per-round time series from bottle-progress.json (CPU)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
root=Path(__file__).resolve().parent
d=json.loads((root/'bottle-progress.json').read_text(encoding='utf-8-sig'))
train=sorted((int(r),row) for r,row in d['train128'].items())
evaluations=sorted((int(r),row) for r,row in d['eval32'].items())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False})
fig,axes=plt.subplots(3,1,figsize=(9.2,6.8),sharex=True)
fig.subplots_adjust(left=.065,right=.978,top=.96,bottom=.085,hspace=.45)
for ax,(key,title,top,ticks) in zip(axes,[('mean_bottles','Bottles per episode',4.08,[0,1,2,3,4]),('mean_official_rubric','Dojo score',103,[0,25,50,75,100]),('success_rate','Full-task success',1.03,[0,.25,.5,.75,1])]):
 ax.set_xlim(-1,train[-1][0]+1)
 ax.set_ylim(0,top);ax.set_yticks(ticks);ax.set_axisbelow(True)
 ax.grid(axis='y',color='#e3e9e5',lw=.7)
 ax.plot([r for r,row in train],[row[key] for r,row in train],color='#7992a6',lw=1.55)
 ax.plot([r for r,row in evaluations],[row[key] for r,row in evaluations],color='#13654e',lw=2.25,marker='o',ms=4.9,markeredgewidth=.8,markeredgecolor='white',zorder=4)
 ax.set_title(title,loc='left',weight='bold',size=12,pad=6,color='#18372e')
 ax.tick_params(axis='both',length=0,color='#e3e9e5',labelcolor='#465953',labelsize=9,pad=3)
 if key=='success_rate':ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[-1].set_xticks(list(range(0,train[-1][0]+1,10)));axes[-1].set_xlabel('Training round',color='#465953',fontsize=10)
fig.savefig(root/'bottle-progress.svg',format='svg',metadata={'Date':None})
print(len(train),'training rounds',len(evaluations),'evaluations')

