"""Complete, prespecified descriptive views; no GPU and no curve smoothing."""
import csv,json,os,tarfile
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.stats import spearmanr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
D=Path('/home/chenyiteng/deployment-20261029/fqlfix-shadow-sz1')
B=Path('/home/chenyiteng/results/pi05-distill/fqlfix-shadow-sz1-20261029/teacher-shadow/batch0')
O=Path('/home/chenyiteng/deployment-20261029/signal-analysis-all-1029');O.mkdir(exist_ok=True)
font_manager.fontManager.addfont(str(D/'msyh.ttc'))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(D/'msyh.ttc')).get_name(),'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.unicode_minus':False})
rows=list(csv.DictReader((B/'signals.csv').open()))
keys=['slot','seed','query','action','u','noise4_u','noise3_u','teacher_student_rmse']
v={k:np.array([float(r[k]) for r in rows]) for k in keys}
v['success']=np.array([r['success']=='True' for r in rows]);slots=np.unique(v['slot']).astype(int)
x=v['noise4_u'];y=v['teacher_student_rmse'];s=v['slot'];q=v['query'];action=v['action']
def rho(a,b):return float(spearmanr(a,b).statistic) if len(a)>2 and np.ptp(a)>0 and np.ptp(b)>0 else None
def export(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(O/(name+'.'+ext),dpi=160,bbox_inches='tight')
 plt.close(fig)
def cluster_ci(a,b,ids,seed=1029,reps=600):
 rng=np.random.default_rng(seed);groups=[np.flatnonzero(ids==i) for i in np.unique(ids)];r=[]
 for _ in range(reps):
  ii=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);r.append(rho(a[ii],b[ii]))
 return np.nanquantile(r,[.025,.975]).tolist()
per=[]
for slot in slots:
 mask=s==slot;xx,yy=x[mask],y[mask];aa=action[mask]
 row=dict(slot=int(slot),seed=int(v['seed'][mask][0]),success=bool(v['success'][mask][0]),actions=int(mask.sum()),
  spearman=rho(xx,yy),spearman_noise3=rho(v['noise3_u'][mask],yy),spearman_original_u=rho(v['u'][mask],yy),
  mean_noise=float(xx.mean()),mean_distance=float(yy.mean()),noise_range=float(np.ptp(xx)),distance_range=float(np.ptp(yy)))
 row['peak_action_noise']=int(aa[np.argmax(xx)]);row['peak_action_distance']=int(aa[np.argmax(yy)])
 per.append(row)
 if not row['success']:continue
 fig,ax=plt.subplots(2,1,figsize=(9.6,4.5),sharex=True,layout='constrained')
 for i,(vals,label,col) in enumerate([(xx,'四噪声动作分散度','#0072B2'),(yy,'师生动作距离 RMSE','#D55E00')]):
  ax[i].plot(aa,vals,color=col,lw=1.8);ax[i].set_ylabel(label,color=col);ax[i].set_ylim(0, (x.max() if i==0 else y.max())*1.04)
  ax[i].grid(alpha=.15);ax[i].axvspan(0,60,color='#d1e7f2',alpha=.28)
  for b in range(20,201,20):ax[i].axvline(b,color='#999',lw=.6,alpha=.35)
 ax[1].set(xlabel='已执行动作序号（每20个动作重新观察）',xlim=(0,200))
 fig.suptitle(f"成功轨迹 {slot:02d} · seed {row['seed']} · {row['actions']}动作 · ρ={row['spearman']:+.3f}")
 export(fig,f'trajectory-{slot:02d}')
# All successes, same axes; no smoothing, no hidden negative examples.
successes=[r for r in per if r['success']]
fig,axs=plt.subplots(4,3,figsize=(15,13),sharex=True,sharey=True,layout='constrained')
for ax,row in zip(axs.flat,successes):
 mask=s==row['slot'];a=action[mask]
 # Scaling is fixed across all trajectories, not independently stretched per episode.
 ax.plot(a,x[mask]/x.max(),c='#0072B2',label='分散度 / 全体最大值')
 ax.plot(a,y[mask]/y.max(),c='#D55E00',ls='--',label='距离 / 全体最大值')
 ax.axvspan(0,60,color='#d1e7f2',alpha=.3);ax.set(xlim=(0,200),ylim=(0,1.04),title=f"slot {row['slot']} · ρ={row['spearman']:+.2f}")
 ax.grid(alpha=.18)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.5,1.025))
fig.supxlabel('已执行动作序号');fig.supylabel('按全体固定最大值缩放（只为同图展示）');export(fig,'all-12-successes')

# Quantile-bin relationships, with trajectory-cluster confidence intervals.
def bins(a,b,ids,n=10):
 edges=np.quantile(a,np.linspace(0,1,n+1));ix=np.clip(np.searchsorted(edges,a,side='right')-1,0,n-1)
 result=[];rng=np.random.default_rng(1029);unique=np.unique(ids)
 # Cluster resampling, retaining each trajectory's internal temporal dependence.
 draws=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=600)
 for k in range(n):
  mask=ix==k;vals=b[mask];weight_sum=[];weight_count=[]
  for sid in unique:
   vv=b[mask & (ids==sid)];weight_sum.append(vv.sum());weight_count.append(len(vv))
  den=draws@np.array(weight_count);num=draws@np.array(weight_sum);good=den>0
  ci=np.quantile(num[good]/den[good],[.025,.975])
  result.append(dict(bin=k+1,n=int(mask.sum()),x=float(np.median(a[mask])),mean=float(vals.mean()),median=float(np.median(vals)),q25=float(np.quantile(vals,.25)),q75=float(np.quantile(vals,.75)),lo=float(ci[0]),hi=float(ci[1])))
 return result
groups={'全部32条':np.ones(len(rows),bool),'成功12条':v['success'],'失败20条':~v['success'],'前3块':q<3,'第4块起':q>=3}
binned={name:bins(x[m],y[m],s[m]) for name,m in groups.items()}
fig,axs=plt.subplots(1,3,figsize=(15,4.2),sharey=True,layout='constrained')
for ax,(name,m) in zip(axs,list(groups.items())[:3]):
 b=binned[name];bx=np.arange(1,11);mu=np.array([r['mean'] for r in b]);lo=np.array([r['lo'] for r in b]);hi=np.array([r['hi'] for r in b])
 ax.plot(bx,mu,'o-',c='#0072B2',label='平均距离');ax.fill_between(bx,lo,hi,color='#0072B2',alpha=.16,label='轨迹重采样95%区间')
 ax.plot(bx,[r['median'] for r in b],'+--',c='#D55E00',label='中位数');ax.set(title=f'{name} · ρ={rho(x[m],y[m]):+.3f}',xlabel='四噪声分散度：等量10档（低→高）',xticks=[1,3,5,7,10],ylim=(0,None));ax.grid(alpha=.15)
axs[0].set_ylabel('师生动作距离 RMSE');axs[0].legend(fontsize=8);export(fig,'quantile-bins')
reverse=bins(y,x,s)
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');bx=np.arange(1,11)
ax.plot(bx,[r['mean'] for r in reverse],'o-',c='#0072B2');ax.fill_between(bx,[r['lo'] for r in reverse],[r['hi'] for r in reverse],color='#0072B2',alpha=.18)
ax.set(xlabel='师生动作距离：等量10档（小→大）',ylabel='平均四噪声分散度',title='反向分箱：给定距离，分散度怎样变化',ylim=(0,None));ax.grid(alpha=.15);export(fig,'reverse-bins')
fig,axs=plt.subplots(1,3,figsize=(15,4.4),layout='constrained',sharex=True,sharey=True)
for ax,(name,m) in zip(axs,list(groups.items())[:3]):
 h=ax.hexbin(x[m],y[m],gridsize=24,mincnt=1,bins='log',cmap='Blues',extent=(0,x.max(),0,y.max()));fig.colorbar(h,ax=ax,label='动作点数（对数色阶）');ax.set(title=name,xlabel='四噪声动作分散度')
axs[0].set_ylabel('师生动作距离 RMSE');export(fig,'density')
# Column-normalized joint histogram: no central-density masking, all data retained.
xe=np.linspace(0,x.max()+1e-9,16);ye=np.linspace(0,y.max()+1e-9,19);hist,_,_=np.histogram2d(x,y,[xe,ye]);conditional=np.divide(hist,hist.sum(axis=1,keepdims=True),out=np.zeros_like(hist),where=hist.sum(axis=1,keepdims=True)>0)
fig,axs=plt.subplots(2,1,figsize=(8,6),height_ratios=[3,1],sharex=True,layout='constrained')
h=axs[0].pcolormesh(xe,ye,conditional.T,cmap='Blues',vmin=0);fig.colorbar(h,ax=axs[0],label='该分散度档内的动作比例')
axs[0].set(ylabel='师生动作距离 RMSE',title='每个分散度区间单独归一化');axs[1].bar(xe[:-1],hist.sum(axis=1),width=np.diff(xe),align='edge',color='#8b9dae');axs[1].set(xlabel='四噪声动作分散度',ylabel='动作数');export(fig,'conditional-density')
# Time/query phase and within-trajectory/within-block associations.
phase=[]
for k in range(10):
 m=q==k;phase.append(dict(query=k+1,actions=int(m.sum()),episodes=int(len(np.unique(s[m]))),rho=rho(x[m],y[m]),noise=float(x[m].mean()),distance=float(y[m].mean())))
fig,axs=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
for name,col in [('前3块','#0072B2'),('第4块起','#D55E00')]:
 b=binned[name];axs[0].plot(range(1,11),[r['mean'] for r in b],'o-',label=name,color=col)
axs[0].set(xlabel='各阶段内分散度：等量10档',ylabel='平均师生距离',ylim=(0,None));axs[0].legend();axs[0].grid(alpha=.15)
axs[1].bar([r['query'] for r in phase],[r['rho'] for r in phase],color='#009E73');axs[1].axhline(0,c='#444',lw=.8);axs[1].set(xlabel='动作块序号',ylabel='该块内 Spearman ρ',ylim=(-1,1),xticks=range(1,11))
ordered=sorted(per,key=lambda r:r['slot']);axs[2].bar([r['slot'] for r in ordered],[r['spearman'] for r in ordered],color=['#009E73' if r['success'] else '#aaa' for r in ordered]);axs[2].axhline(0,c='#444',lw=.8);axs[2].set(xlabel='轨迹 slot（绿=成功，灰=失败）',ylabel='整条轨迹 Spearman ρ',ylim=(-1,1));export(fig,'phase-and-trajectories')
def centered_rank_corr(group):
 a=rankdata(x);b=rankdata(y)
 for k in np.unique(group):
  m=group==k;a[m]-=a[m].mean();b[m]-=b[m].mean()
 return float(np.corrcoef(a,b)[0,1])
# Retention/sparsification view (disagreement only, not verified action error).
fractions=np.linspace(0,.9,19);risk={}
for key,order in [('noise4',np.argsort(x)),('noise3',np.argsort(v['noise3_u'])),('original_u',np.argsort(v['u'])),('oracle_distance',np.argsort(y))]:
 risk[key]=[float(y[order[:max(1,round(len(y)*(1-f)))]] .mean()/y.mean()) for f in fractions]
fig,ax=plt.subplots(figsize=(8.5,4.7),layout='constrained')
for key,label,col,style in [('noise4','按四噪声分散度排除','#0072B2','-'),('noise3','按另3噪声分散度排除','#56B4E9','--'),('original_u','按原U排除（补充）','#009E73','-.'),('oracle_distance','直接按距离排除（理想排序）','#D55E00',':')]:
 ax.plot(fractions*100,risk[key],label=label,c=col,ls=style,lw=2)
ax.axhline(1,c='#777',ls='--',label='随机排除的期望');ax.set(xlabel='先排除高信号动作的比例（%）',ylabel='剩余平均距离 / 原平均距离',ylim=(0,None),title='能否挑出师生分歧大的动作？');ax.legend(fontsize=9);ax.grid(alpha=.15);export(fig,'sparsification')
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for success,col,lab in [(True,'#009E73','成功'),(False,'#777','失败')]:
 m=v['success']==success;axs[0].hist(x[m],bins=np.linspace(0,x.max(),30),histtype='step',density=True,color=col,label=lab,lw=2)
 rr=[r for r in per if r['success']==success];axs[1].scatter([r['mean_noise'] for r in rr],[r['mean_distance'] for r in rr],color=col,label=lab)
 for r in rr:axs[1].annotate(str(r['slot']),(r['mean_noise'],r['mean_distance']),fontsize=7)
axs[0].set(xlabel='四噪声动作分散度',ylabel='概率密度',title='分散度的分布');axs[1].set(xlabel='每条轨迹平均分散度',ylabel='每条轨迹平均师生距离',title='每条轨迹权重相同');axs[0].legend();axs[1].legend();export(fig,'distributions')
stats={name:dict(actions=int(m.sum()),episodes=int(len(np.unique(s[m]))),rho=rho(x[m],y[m]),ci=cluster_ci(x[m],y[m],s[m])) for name,m in groups.items()}
summary=dict(episodes=32,successes=len(successes),actions=len(rows),groups=stats,per_episode=per,phase=phase,
 noise3_rho=rho(v['noise3_u'],y),original_u_rho=rho(v['u'],y),within_episode_rank_corr=centered_rank_corr(s),within_episode_query_rank_corr=centered_rank_corr(s*100+q),
 episode_mean_rho=rho(np.array([r['mean_noise'] for r in per]),np.array([r['mean_distance'] for r in per])),
 binned=binned,reverse_binned=reverse,sparsification=dict(fraction_removed=fractions.tolist(),curves=risk),
 definitions={'noise4_u':'std of 4 noise-conditioned predictions / coordinate RMS, averaged over physical 14 dimensions; not entropy',
 'distance':'normalized 14-dimensional RMSE, weak z0 versus strong z0 at same observation; not ground-truth error'},
 selection='all 12 successes, fixed global scales, no smoothing; no subset selected for a positive relationship',
 bootstrap='600 trajectory-cluster resamples; descriptive interval, not causal or independent-action significance')
(O/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
with (O/'trajectory-statistics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
with tarfile.open(D/'signal-analysis-all.tar','w') as t:
 for p in O.iterdir():t.add(p,arcname=p.name)
print(json.dumps({k:summary[k] for k in ['episodes','successes','actions','groups','noise3_rho','original_u_rho','within_episode_rank_corr','within_episode_query_rank_corr','episode_mean_rho']},ensure_ascii=False))
