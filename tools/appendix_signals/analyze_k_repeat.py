import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import json,itertools,argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--prior-analysis',required=True);p.add_argument('--output',required=True);args=p.parse_args();D=Path(args.data);O=Path(args.output);P=O/'public';P.mkdir(parents=True,exist_ok=True)
old=json.loads(Path(args.prior_analysis).read_text());counts=old['K_definition']['executed_counts'];pool=json.loads((D/'rt-pool/queries.json').read_text())
banks=[np.load(D/f'rt-k8/bank_{i}.npz')['model_actions'].astype(float) for i in [0,1]]
ent=lambda x:.5*np.log(2*np.pi*np.e*(x.var(axis=1,ddof=1)+1e-6)).sum(-1)
rows=[];examples=[]
for k in [2,3,4,8]:
 subsets=list(itertools.combinations(range(8),k));byobs=[];overlaps=[]
 for subset in subsets:
  hs=[ent(b[:,subset]) for b in banks];values=[];ov=[]
  for i,n in enumerate(counts):
   if n<5:values.append(np.nan);ov.append(np.nan);continue
   a,b=[rankdata(h[i,:n]) for h in hs];values.append(float(np.corrcoef(a,b)[0,1]));m=int(np.ceil(n*.2));ov.append(len(set(np.argsort(a)[-m:])&set(np.argsort(b)[-m:]))/m)
  byobs.append(values);overlaps.append(ov)
 per=np.nanmean(byobs,axis=0);perov=np.nanmean(overlaps,axis=0);groups={}
 for i,q in enumerate(pool):
  if np.isfinite(per[i]):groups.setdefault((q['task'],q['slot']),[]).append(per[i])
 ep=np.array([np.mean(x) for x in groups.values()]);rng=np.random.default_rng(1126);ci=np.quantile(rng.choice(ep,size=(2000,len(ep)),replace=True).mean(1),[.025,.975])
 rows.append({'K':k,'repeat_rho':float(ep.mean()),'ci95':ci.tolist(),'top20_overlap':float(np.nanmean(perov)),'comparisons_per_observation':len(subsets),'per_obs':[float(x) if np.isfinite(x) else None for x in per]})
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(1,2,figsize=(9.8,3.1));ks=[r['K'] for r in rows];rho=np.array([r['repeat_rho'] for r in rows]);ci=np.array([r['ci95'] for r in rows]);ax[0].errorbar(ks,rho,yerr=[rho-ci[:,0],ci[:,1]-rho],fmt='o-',capsize=4,color='#559fda');ax[0].set(xlabel='Noise samples per independent run (K)',ylabel='Action-ranking agreement',ylim=(0,1),xticks=ks);ax[1].plot(ks,[r['top20_overlap']*100 for r in rows],'o-',color='#e77a64');ax[1].axhline(np.mean([np.ceil(n*.2)/n for n in counts if n>=5])*100,color='gray',ls='--',label='Random choice');ax[1].set(xlabel='K samples per run',ylabel='High-entropy actions shared (%)',ylim=(0,100),xticks=ks);ax[1].legend(fontsize=8)
for a in ax:a.grid(alpha=.14)
fig.tight_layout();fig.savefig(P/'k_repeat.png',dpi=180,bbox_inches='tight');fig.savefig(P/'k_repeat.svg',bbox_inches='tight');plt.close(fig)
# A fixed median-agreement example, showing two independent K4 repeats.
vals=rows[2]['per_obs'];ids=[i for i,v in enumerate(vals) if v is not None];med=np.median([vals[i] for i in ids]);i=min(ids,key=lambda j:abs(vals[j]-med));n=counts[i];hh=[ent(b[i:i+1,:4])[0,:n] for b in banks]
fig,ax=plt.subplots(figsize=(9.6,2.7))
for h,color,name in zip(hh,['#559fda','#e77a64'],['Independent K4 sample A','Independent K4 sample B']):ax.plot(range(1,n+1),h,'o-',ms=3,lw=1.2,color=color,label=name)
ax.set(xlabel='Action position in the same observation',ylabel='Action entropy (nats)');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.13);fig.tight_layout();fig.savefig(P/'k_repeat_example.png',dpi=180,bbox_inches='tight');fig.savefig(P/'k_repeat_example.svg',bbox_inches='tight');plt.close(fig)
out={'rows':rows,'same_k_repeats':True,'observations':62,'episodes':16,'sample_pool_observations':64,'definition':'For each fixed observation, compare K independent samples from bank A with K from independent bank B. Enumerate the same index subsets in independent banks, average within observation then episode.','reference_k8_analysis':old['K'],'example':{'index':i,'task':pool[i]['task'],'slot':pool[i]['slot'],'query':pool[i]['query'],'entropy':np.array(hh).tolist()}}
(O/'k-repeat.json').write_text(json.dumps(out,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='per_obs'} for r in rows]))
