import csv,json,os,sys,itertools
from pathlib import Path
from datetime import datetime,timezone,timedelta
BASE=Path(__file__).resolve().parents[1];ROOT=Path('E:/Codex/home/visualizations/2026/10/06/01a10f72-dbcf-70e1-8236-beac311711d1');DEST=ROOT/'dsrl-history-review';DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'plot_deps'));os.environ['MPLCONFIGDIR']=str(DEST/'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'font.size':11,'axes.spines.right':False,'axes.spines.top':False,'axes.edgecolor':'#cad4d4'})
oldroot=Path('C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-rlt-dsrl-port/evidence/formal-artifact-refresh-20260824')
hist=list(csv.DictReader((oldroot/'dsrl_v2_metrics_step1_200.csv').open(encoding='utf-8-sig')))
hev=list(csv.DictReader((oldroot/'dsrl_v2_fixed12.csv').open(encoding='utf-8-sig')))
now=json.loads((BASE/'evidence/p089-dsrl-phase.out').read_text());stamp=datetime.fromtimestamp(now['time'],timezone(timedelta(hours=8))).strftime('%m-%d %H:%M')
colors={'history':'#899799','clean':'#176b70','u':'#e88a32'};labels={'history':'历史 π0 · 8月24日','clean':'当前 π0.5 · Clean','u':'当前 π0.5 · ＋U'}
fig=plt.figure(figsize=(14.5,8.8));fig.patch.set_facecolor('#f6f8f8');gs=fig.add_gridspec(2,2,left=.065,right=.97,bottom=.13,top=.79,hspace=.5,wspace=.22);ax=fig.add_subplot(gs[0,:]);a=fig.add_subplot(gs[1,0]);b=fig.add_subplot(gs[1,1])
fig.text(.065,.945,'DSRL  |  历史阶段与当前对照',fontsize=24,weight='bold',color='#17383c')
fig.text(.065,.89,f'当前快照 {stamp} 北京时间 · 同为 adjust_bottle；模型、求解步数、动作块和并行配置不同',color='#63777b',fontsize=12)
for q in (ax,a,b):q.set_facecolor('white');q.grid(axis='y',alpha=.2);q.set_axisbelow(True);q.set_xlabel('训练轮次')
ax.plot([int(q['runner_step']) for q in hev],[float(q['success'])*100 for q in hev],color=colors['history'],marker='o',lw=2,label=labels['history'])
for name in ('clean','u'):
 m=now['runs'][name]['metrics'];v=m['eval/success_once'];ax.plot([q['step']+1 for q in v],[100*q['value'] for q in v],color=colors[name],marker='o',lw=2,label=labels[name])
ax.set_title('每次 12 条随机评估｜历史第65轮已12/12；当前第65轮均0/12',loc='left',weight='bold',pad=13);ax.set_ylabel('成功率（%）');ax.set_ylim(-4,113);ax.set_xlim(0,205);ax.legend(loc='lower right',frameon=False,ncol=3,fontsize=10)
ax.annotate('历史最终 8/12',(200,66.6667),xytext=(-97,-29),textcoords='offset points',color=colors['history'])
def rolling(y,n=10):return [sum(y[i-n+1:i+1])/n*100 for i in range(n-1,len(y))]
hy=[float(q['train_success']) for q in hist];a.plot(range(10,201),rolling(hy),color=colors['history'],lw=2)
hx=[int(q['step']) for q in hist];b.plot(hx,[float(q['cumulative_optimizer_updates'])/1000 for q in hist],color=colors['history'],lw=2)
summary={}
for name in ('clean','u'):
 m=now['runs'][name]['metrics'];v=m['env/success_once'];a.plot([q['step']+1 for q in v[9:]],rolling([q['value'] for q in v]),color=colors[name],lw=2)
 v=m['train/sac/planned_optimizer_updates'];updates=list(itertools.accumulate(q['value'] for q in v));b.plot([q['step']+1 for q in v],[q/1000 for q in updates],color=colors[name],lw=2)
 summary[name]={'round':m['env/success_once'][-1]['step']+1,'first_update_round':m['train/sac/critic_loss'][0]['step']+1,'cumulative_planned_updates':updates[-1],'replay':m['train/sac/global_resident_transitions'][-1]['value']}
a.set_title('在线采集｜最近10轮均值，独立于评估',loc='left',weight='bold',pad=13);a.set_ylabel('采集成功率（%）');a.set_ylim(0,105);a.set_xlim(0,205)
b.set_title('累计计划更新｜当前已充分越过预热',loc='left',weight='bold',pad=13);b.set_ylabel('优化器更新（千次）');b.set_xlim(0,205);b.set_ylim(bottom=0)
fig.text(.065,.065,'阶段：历史 R1–12 收集500条回放，R13开始SAC；当前 R1–6预采集，R7已开始SAC，无RLT式额外15000次初始化。',color='#63777b',fontsize=10.5)
fig.text(.065,.025,'判断：当前进程/梯度正常，但学习效果明显偏弱；不能用“仍在预热”解释，也不能将跨模型差距直接归因于U。',color='#63777b',fontsize=10.5)
fig.savefig(DEST/'dsrl_history_vs_current.png',dpi=150,facecolor=fig.get_facecolor());fig.savefig(DEST/'dsrl_history_vs_current.svg',facecolor=fig.get_facecolor())
(DEST/'stage_summary.json').write_text(json.dumps(summary,indent=2));(DEST/'current_snapshot.json').write_text(json.dumps(now,indent=2))
print(json.dumps({'figure':str(DEST/'dsrl_history_vs_current.png'),'summary':summary}))
