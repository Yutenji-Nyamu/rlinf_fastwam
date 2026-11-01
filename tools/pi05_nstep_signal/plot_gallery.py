"""CPU figures based on accepted U gallery layout; no smoothing or frame guesses."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
O=Path(sys.argv[1]);items=json.loads((O/'manifest.json').read_text())['selected']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
groups=[('budget','Same-noise solver disagreement',[('u_1_10','N1 vs N10','#D55E00'),('u_1_5','N1 vs N5','#0072B2'),('u_5_10','N5 vs N10','#009E73')]),('rmse','Model-coordinate action RMSE',[('rmse_1_10','N1 vs N10','#D55E00'),('rmse_1_5','N1 vs N5','#0072B2'),('rmse_5_10','N5 vs N10','#009E73')]),('noise_abs','K4 absolute dispersion (mean std)',[(f'noise4_absolute_n{n}',f'N{n}',color)for n,color in [(1,'#0072B2'),(5,'#7B2CBF'),(10,'#009E73')]]),('noise_rel','K4 relative dispersion',[(f'noise4_relative_n{n}',f'N{n}',color)for n,color in [(1,'#0072B2'),(5,'#7B2CBF'),(10,'#009E73')]]),('dv','DV of endpoint estimates (N1 unavailable)',[(f'dv_n{n}',f'N{n}',color)for n,color in [(5,'#7B2CBF'),(10,'#009E73')]])]
def panel(ax,c,group,annotate=False):
 x=np.asarray(c['action']);gap=np.flatnonzero(np.diff(x)>1)+1
 for k,label,color in group[2]:
  xx=np.insert(x.astype(float),gap,np.nan);yy=np.insert(np.asarray(c['signals'][k]),gap,np.nan)
  ax.plot(xx,yy,color=color,label=label,lw=1.25)
 for idx in np.flatnonzero(np.diff(c['query'])!=0)+1:ax.axvline(x[idx],color='#B0B8C0',lw=.5,ls=':')
 for letter,p in zip('ABCDEF',c['options']):
  ax.axvline(p['action'],color='#7A7A7A',lw=.6,ls='--',alpha=.6)
  if annotate:ax.text(p['action'],1.01,letter,transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
 labels={'budget':'Solver gap\n(relative)','rmse':'Action RMSE','noise_abs':'K4 std\n(absolute)','noise_rel':'K4 dispersion\n(relative)','dv':'DV (N1 missing)'}
 ax.set_xlim(x[0],x[-1]);ax.set_ylim(bottom=0);ax.grid(alpha=.13);ax.set_ylabel(labels[group[0]]);ax.legend(loc='upper right',fontsize=8,ncol=3)
for c in items:
 d=O/c['id'];fig,axes=plt.subplots(5,1,figsize=(13.8,10.6),sharex=True,layout='constrained')
 for j,g in enumerate(groups):panel(axes[j],c,g,j==0)
 axes[-1].set_xlabel('Action index (controller-projected executed prefix)')
 fig.suptitle(f"{c['id']} | {c['task']} | seed {c['actual']} | "+('SUCCESS'if c['success']else'FAILURE'),fontsize=13)
 for suffix in ['svg','png','pdf']:fig.savefig(d/f'curve.{suffix}',dpi=125)
 plt.close(fig)
 for g in groups:
  fig,ax=plt.subplots(figsize=(13.8,2.35),layout='constrained');panel(ax,c,g,True);ax.set_xlabel('Action index');fig.savefig(d/f'{g[0]}.svg');plt.close(fig)
 # Paper-ready contact sheet with six real frames and exact provenance labels.
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
 contact=Image.new('RGB',(1440,650),'white');draw=ImageDraw.Draw(contact)
 for j,p in enumerate(c['options']):
  pic=Image.open(d/p['image']).convert('RGB');pic.thumbnail((468,275));x=(j%3)*480;y=(j//3)*325;contact.paste(pic,(x,y))
  draw.text((x+5,y+280),f"{'ABCDEF'[j]} action {p['action']} | frame {p['frame']}",font=font,fill='#183348')
  draw.text((x+5,y+302),f"query {p['query']} | projection error {p['h_error']:.2f}",font=font,fill='#536577')
 contact.save(d/'contact.jpg',quality=92)
 print(c['id'],flush=True)
# Compact visual QA sheets: every chosen trajectory, first/peak/last frame.
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
for start in range(0,len(items),8):
 group=items[start:start+8];im=Image.new('RGB',(1200,len(group)*210),'white');draw=ImageDraw.Draw(im)
 for row,c in enumerate(group):
  y=row*210;draw.text((5,y+2),f"{c['id']} {c['task']} | seed {c['actual']} | {'SUCCESS'if c['success']else'FAILURE'}",font=font,fill='#183348')
  opts=[c['options'][0],min(c['options'],key=lambda p:abs(p['index']-c['peak_index'])),c['options'][-1]]
  for col,p in enumerate(opts):
   pic=Image.open(O/c['id']/p['image']).convert('RGB');pic.thumbnail((390,170));im.paste(pic,(col*400,y+27));draw.text((col*400+4,y+192),f"action {p['action']}",font=font,fill='#536577')
 im.save(O/f'qa-{start//8+1:02d}.jpg',quality=90)
(O/'plots-complete.json').write_text(json.dumps({'trajectories':len(items),'groups':[x[0]for x in groups]}))
