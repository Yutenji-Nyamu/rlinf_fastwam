import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();O=Path(a.data);rows=json.loads((O/'curves.json').read_text());plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
for g,r in rows.items():
 fig,ax=plt.subplots(figsize=(10.6,3.3))
 for i,s in reversed(list(enumerate(r['series']))):
  p=s['eval'];ax.plot([x[0] for x in p],[x[1] for x in p],'o-',ms=3,color=['#237e98','#8d9da5','#c7cfd2'][i],label=['Last setting / resume segment','Previous segment','Earlier segment'][i])
 ax.set(title=r['title']+' | historical run',xlabel='Logged round (settings changed between segments)',ylabel='Evaluation success (%)',ylim=(-5,105));ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=3,loc='lower left');fig.tight_layout()
 for ext in ['png','svg','pdf']:fig.savefig(O/'public'/('training_'+g+'.'+ext),dpi=170)
 plt.close(fig)
print('rendered 3 historical families')
