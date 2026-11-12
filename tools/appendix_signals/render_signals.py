"""Render saved action-level series and frame references. CPU only."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(root):
 root=Path(root);rows=json.loads((root/'signals.json').read_text())
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','font.family':'DejaVu Sans'})
 for row in rows:
  folder=root/row['visibility'];folder.mkdir(exist_ok=True);fig=plt.figure(figsize=(10.6,4.35));gs=fig.add_gridspec(2,3,height_ratios=[1.45,1],hspace=.28,wspace=.05)
  for j,frame in enumerate(row['frames']):
   ax=fig.add_subplot(gs[0,j]);ax.imshow(Image.open(root/frame['path']));ax.axis('off');ax.set_title(f"{'ABC'[j]}  |  {frame['label']}",fontsize=10,color='#304650',pad=5)
  ax=fig.add_subplot(gs[1,:]);x=np.asarray(row['x']);y=np.asarray(row['entropy']);chunks=row['chunks']
  for j,c in enumerate(chunks):
   if j%2==0:ax.axvspan(c['start'],c['end'],color='#dce8ed',alpha=.35,lw=0)
  ax.plot(x,y,color='#2480b5',lw=1.35)
  for j,f in enumerate(row['frames']):
   idx=int(np.argmin(abs(x-f['x'])));ax.axvline(x[idx],ls=':',lw=.8,color='#9badb7');ax.scatter(x[idx],y[idx],s=20,c='#c97355',zorder=4);ax.annotate('ABC'[j],(x[idx],y[idx]),xytext=(4,7),textcoords='offset points',fontsize=9,color='#ad5c42')
  ax.set(xlabel=row['xlabel'],ylabel='Action entropy (nats)',xlim=(x[0],x[-1]));ax.grid(axis='y',alpha=.13);fig.suptitle(row['title'],fontsize=12,y=.99)
  fig.subplots_adjust(left=.075,right=.99,bottom=.115,top=.885)
  for ext in ['png','svg','pdf']:fig.savefig(folder/(row['id']+'.'+ext),dpi=170)
  plt.close(fig)
  with (folder/(row['id']+'.csv')).open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['action_or_time','entropy_nats','chunk']);w.writerows(zip(x,y,row['query']))
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();render(a.data)
