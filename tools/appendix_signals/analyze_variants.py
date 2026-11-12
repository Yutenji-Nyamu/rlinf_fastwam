import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import argparse,json,csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(old,data,out):
 old=Path(old);data=Path(data);out=Path(out);p=out/'public';p.mkdir(exist_ok=True)
 read=lambda f:json.loads(Path(f).read_text());complete=read(data/'complete.json');nodes=read(data/'nodes.json');z=np.load(data/'embeddings.npz');emb=z['per_view'];grip=z['gripper_closed'];lookup={(n['pair'],n['variant']):i for i,n in enumerate(nodes)}
 on=read(old/'nodes.json');triplets=read(old/'triplets.json');rows=[]
 for t in triplets:
  an=on[t['anchor']];cn=on[t['other_time_same_light']];a=lookup[(an['pair'],'base')];c=lookup[(cn['pair'],'base')];assert np.array_equal(grip[a],grip[c]);den=float(np.linalg.norm(emb[a]-emb[c],axis=-1).mean())
  for variant in complete['variants'][1:]:
   if (an['pair'],variant) not in lookup:continue
   b=lookup[(an['pair'],variant)];num=float(np.linalg.norm(emb[a]-emb[b],axis=-1).mean());rows.append({'task':an['task'],'seed':an['seed'],'anchor':a,'variant_node':b,'other_stage':c,'variant':variant,'distance_appearance':num,'distance_stage':den,'ratio':num/max(den,1e-9),'appearance_farther':num>den,'anchor_query':an['query'],'other_query':cn['query']})
 stats=[]
 for v in complete['variants'][1:]:
  rs=[r for r in rows if r['variant']==v];stats.append({'variant':v,'triplets':len(rs),'farther':sum(r['appearance_farther'] for r in rs),'median_ratio':float(np.median([r['ratio'] for r in rs])),'episodes':len({(r['task'],r['seed']) for r in rs})})
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 def save(fig,name):
  for ext in ['png','svg','pdf']:fig.savefig(p/(name+'.'+ext),dpi=170,bbox_inches='tight')
  plt.close(fig)
 labels={'warm_light':'Warm light','dim_light':'Dim light','camera_5cm':'Camera +5 cm','texture_1':'Texture 1','texture_2':'Texture 2',**{f'native_rand_{i}':f'Native rand {i+1}' for i in range(4)}}
 fig,ax=plt.subplots(figsize=(10,3.8))
 for j,s in enumerate(stats):
  values=np.array([r['ratio'] for r in rows if r['variant']==s['variant']]);xx=j+np.linspace(-.16,.16,len(values));ax.scatter(xx,values,s=17,color='#3d8b9f',alpha=.6);ax.plot([j-.23,j+.23],[s['median_ratio']]*2,color='#d47b53',lw=2.5)
  ax.text(j,.03,f"{s['farther']}/{s['triplets']} > 1",transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
 ax.axhline(1,ls='--',color='#7c8d96',lw=1);ax.set(xticks=range(len(stats)),xticklabels=[labels[s['variant']] for s in stats],ylabel='Appearance distance / pose-stage distance',title='Same state, changed appearance: all fixed triplets retained',yscale='log');ax.grid(axis='y',alpha=.15);save(fig,'grbc_variants_summary')
 selected=[]
 # Reuse the already visually reviewed bowls-separated -> near-stacking example.
 for v in complete['variants'][1:]:
  r=next(r for r in rows if r['task']=='stack_bowls_two' and r['seed']==100100035 and r['anchor_query']==0 and r['other_query']==6 and r['variant']==v);selected.append(r)
  fig=plt.figure(figsize=(10.5,4.5));gs=fig.add_gridspec(2,3,height_ratios=[2,1],hspace=.3)
  for j,key in enumerate(['anchor','variant_node','other_stage']):
   node=nodes[r[key]];im=np.load(data/node['raw'])['images'][node['appearance'],0];ax=fig.add_subplot(gs[0,j]);ax.imshow(im);ax.axis('off');ax.set_title(['A: bowls separated','B: same state / '+labels[v],'C: near stacking'][j],fontsize=10)
  ax=fig.add_subplot(gs[1,:]);vals=[r['distance_stage'],r['distance_appearance']];ax.barh(['A to C: changed pose','A to B: changed appearance'],vals,color=['#3e95a2','#d98d57']);ax.set(xlabel='Mean L2 distance of normalized three-view MCR features',xlim=(0,max(vals)*1.22));ax.grid(axis='x',alpha=.15)
  for j,vv in enumerate(vals):ax.text(vv+max(vals)*.02,j,f'{vv:.3f}',va='center')
  save(fig,'grbc_'+v)
 with (p/'grbc_variants.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result={'collection':complete,'statistics':stats,'triplets':rows,'selected':selected,'scope':'Fixed 8-episode appearance diagnostic, using old same-gripper triplet pool. Repeated nodes are dependent. Ratios below one are retained. Not policy success-rate testing.'};(out/'grbc-variants-analysis.json').write_text(json.dumps(result));print(json.dumps({'collection':complete,'statistics':stats,'selected':selected}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.old,a.data,a.output)
