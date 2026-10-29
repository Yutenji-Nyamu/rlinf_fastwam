import csv,json,os,tarfile
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.stats import spearmanr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
D=Path('/home/chenyiteng/deployment-20261029/fqlfix-shadow-sz1');O=Path('/home/chenyiteng/deployment-20261029/signal-analysis-all-1029')
B=Path('/home/chenyiteng/results/pi05-distill/fqlfix-shadow-sz1-20261029/teacher-shadow/batch0')
font_manager.fontManager.addfont(str(D/'msyh.ttc'));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(D/'msyh.ttc')).get_name();plt.rcParams['axes.unicode_minus']=False
rows=list(csv.DictReader((B/'signals.csv').open()));v={k:np.array([float(r[k]) for r in rows]) for k in ['noise4_u','teacher_student_rmse','slot','query','action']}
x=v['noise4_u'];y=v['teacher_student_rmse'];group=v['slot']*100+v['query'];offset=v['action']%20
def remove(a,groups):
 r=a.copy()
 for k in np.unique(groups):
  m=groups==k;r[m]-=a[m].mean()
 return r
rx=remove(rankdata(x),group);ry=remove(rankdata(y),group);rrx=remove(rx,offset);rry=remove(ry,offset)
summ=json.loads((O/'analysis.json').read_text());summ['within_block_and_position_rank_corr']=float(np.corrcoef(rrx,rry)[0,1])
fig,ax=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
ax[0].hexbin(rx,ry,gridsize=26,mincnt=1,bins='log',cmap='Blues');ax[0].set(xlabel='分散度排名－该块均值',ylabel='距离排名－该块均值',title=f"同一动作块内 · r={summ['within_episode_query_rank_corr']:.3f}")
ax[1].hexbin(rrx,rry,gridsize=26,mincnt=1,bins='log',cmap='Blues');ax[1].set(xlabel='再扣除块内位置平均差异',ylabel='距离排名残差',title=f"控制块和位置 · r={summ['within_block_and_position_rank_corr']:.3f}")
xx=np.array([x[offset==h].mean() for h in range(20)]);yy=np.array([y[offset==h].mean() for h in range(20)])
ax[2].plot(range(1,21),xx/xx.mean(),c='#0072B2',label='分散度 / 均值');ax[2].plot(range(1,21),yy/yy.mean(),c='#D55E00',ls='--',label='距离 / 均值');ax[2].set(xlabel='一个动作块内的第几个动作',ylabel='相对各自均值',title='两者是否都随预测位置上升？');ax[2].legend(fontsize=8)
for a in ax:a.grid(alpha=.15)
for ext in ['png','svg','pdf']:fig.savefig(O/('within-block.'+ext),dpi=160,bbox_inches='tight')
plt.close(fig)
# Supplemental absolute dispersion: specified before inspecting its correlation.
# Keep the original relative metric as the primary result; same physical coordinates as RMSE.
files=sorted(B.glob('query_*.npz'))+sorted(B.glob('query-*.npz'))
print('query files', [p.name for p in files[:2]])
arrays={}
for p in files:
 with np.load(p) as z:
  if 'noise4_model_actions' not in z:continue
  k=int(''.join(c for c in p.stem if c.isdigit()));arrays[k]=z['noise4_model_actions']
print('array shapes', {k:a.shape for k,a in arrays.items()})
if arrays:
 raw=[];raw3=[]
 for r in rows:
  a=arrays[int(r['query'])][int(r['slot']),:,int(r['action'])%20,:14].astype(float)
  raw.append(float(np.sqrt(np.mean(np.var(a,axis=0)))))
  raw3.append(float(np.sqrt(np.mean(np.var(a[1:],axis=0)))))
 raw=np.array(raw);raw3=np.array(raw3)
 summ['supplement_absolute_dispersion']={'definition':'RMS of coordinate std across 4 predictions; normalized physical 14-dimensional model coordinates; no RMS denominator; exploratory secondary metric','rho':float(spearmanr(raw,y).statistic),'rho_excluding_main_noise':float(spearmanr(raw3,y).statistic)}
 fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for a,z,label in [(ax[0],x,'原指标：相对分散度'),(ax[1],raw,'补充：绝对分散度')]:
  a.hexbin(z,y,gridsize=30,mincnt=1,bins='log',cmap='Blues');a.set(xlabel=label,ylabel='师生动作距离 RMSE',title=f"全部32条 · ρ={spearmanr(z,y).statistic:+.3f}")
 for ext in ['png','svg','pdf']:fig.savefig(O/('relative-vs-absolute.'+ext),dpi=160,bbox_inches='tight')
 plt.close(fig)
 with (O/'absolute-dispersion.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['slot','query','action','absolute_noise4','absolute_noise3'])
  w.writerows((r['slot'],r['query'],r['action'],float(a),float(b)) for r,a,b in zip(rows,raw,raw3))
(O/'analysis.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2))
with tarfile.open(D/'signal-analysis-all.tar','w') as t:
 for p in O.iterdir():t.add(p,arcname=p.name)
print(json.dumps({k:summ[k] for k in ['within_block_and_position_rank_corr','supplement_absolute_dispersion'] if k in summ},ensure_ascii=False))
