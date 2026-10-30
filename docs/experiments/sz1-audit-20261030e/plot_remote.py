import os,json,pathlib,tarfile
os.environ['CUDA_VISIBLE_DEVICES']=''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=pathlib.Path('/home/chenyiteng/deployment-20261030/queue-storage-sz1')
v=json.loads((D/'plot_input.json').read_text());rows=v['rows'];stamp=v['stamp']
colors={'Direct':'#0072B2','CP':'#E69F00','DMD new':'#009E73','MFD':'#7B2CBF','CP v2':'#D55E00','DMD v2':'#333333','MFD v2':'#CC79A7','Old OneDP (stopped)':'#009E73','Old OneDP':'#009E73'}
markers=['o','s','^','D','*','X','P']
if plt is not None:
 from matplotlib import font_manager
 candidates=[p for p in font_manager.findSystemFonts() if any(s in p.lower() for s in ['notosanscjk','wqy','droidsansfallback','msyh'])]
 candidates.insert(0,'/home/chenyiteng/deployment-20261029/fqlfix-shadow-sz1/msyh.ttc')
 if candidates:
  font_manager.fontManager.addfont(candidates[0]);plt.rcParams['font.family']=font_manager.FontProperties(fname=candidates[0]).get_name()
for key,protocol,ylim in [('native','Native single task',(0,100)),('c10','C10 multitask',(0,50)),('c50','C50 multitask history',(35,60))]:
 if plt is None:continue
 fig,ax=plt.subplots(figsize=(12,5.5),layout='constrained')
 subset=[r for r in rows if r['protocol']==protocol and r['complete'] and 'FQL' not in r['method']]
 methods=list(dict.fromkeys([n for n in colors if any(r['method']==n for r in subset)]))
 for i,m in enumerate(methods):
  points=sorted([r for r in subset if r['method']==m],key=lambda r:r['step'])
  per_epoch=({'Direct':11,'CP':30,'CP v2':30,'DMD new':13,'MFD':13,'DMD v2':13,'MFD v2':13}[m] if key=='native' else 1113)
  xs=[r['step']/per_epoch for r in points];ys=[r['rate'] for r in points]
  if key=='native':xs=[0]+xs;ys=[76.5625]+ys
  ax.plot(xs,ys,color=colors[m],marker=markers[i%len(markers)],ms=6,lw=2,ls='--' if 'v2' in m else '-',label=m)
 refs={'native':[(86.71875,'Original N5'),(76.5625,'Original N1')],'c10':[(153/368*100,'Original N10'),(75/368*100,'Original N1')],'c50':[(193/368*100,'Original N10'),(191/368*100,'Original N1')]}[key]
 for j,(y,n) in enumerate(refs):ax.axhline(y,label=f'{n}: {y:.2f}%',color=['#555555','#999999'][j],ls=['--',':'][j],lw=1.3)
 ax.set(xlabel='训练轮数',ylabel='成功率 (%)',ylim=ylim,title=f'{protocol} | {stamp} CST')
 if key=='native':ax.set_xlim(0,max(200,max(r['step']/({'Direct':11,'CP':30,'CP v2':30,'DMD new':13,'MFD':13,'DMD v2':13,'MFD v2':13}[r['method']]) for r in subset)));ax.set_xticks(range(0,int(ax.get_xlim()[1])+1,25 if ax.get_xlim()[1]<=200 else 50))
 ax.grid(alpha=.18);ax.legend(fontsize=9,ncol=3)
 fig.savefig(D/f'{key}.png',dpi=145);fig.savefig(D/f'{key}.svg');plt.close(fig)

with tarfile.open(D/'plots.tar','w') as t:
 for k in ['native','c10','c50']:
  for ext in ['png','svg']:t.add(D/f'{k}.{ext}',arcname=f'{k}.{ext}')
print('plots ready')

