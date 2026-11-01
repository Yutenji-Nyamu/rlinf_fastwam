import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/home/chenyiteng/deployment-20261101/gad-k4-v1/fql-analysis')
d=json.loads((O/'analysis.json').read_text());p=d['points'];x=[r['epoch']for r in p]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(2,2,figsize=(13.6,8.4),constrained_layout=True)
for ax in axs.flat:
 ax.set_xlim(0,400);ax.set_xticks([0,100,200,300,400]);ax.set_xlabel('Training budget epochs');ax.grid(alpha=.18)
a=axs[0,0];a.plot(x,[r['successes']/r['episodes']*100 for r in p],'-o',color='#0072B2',markersize=4,label='FQL-fix')
a.axhline(86.71875,color='#444',ls='--',lw=1,label='Original N5: 86.7%');a.axhline(76.5625,color='#999',ls=':',label='Original N1: 76.6%');a.set(ylim=(0,100),ylabel='Success rate (%)',title='A. Real dip and recovery (128 episodes / point)');a.legend(fontsize=9,loc='lower right');a.annotate('160: 15.6%',xy=(160,15.625),xytext=(85,5),arrowprops={'arrowstyle':'->','color':'#444'});a.annotate('400: 88.3%',xy=(400,88.28),xytext=(285,96))
a=axs[0,1];a.plot(x,[r['training']['endpoint_mse']for r in p],'-s',color='#D55E00',markersize=4);a.set(ylabel='Normalized action MSE',title='B. Student departs, then moves closer to teacher');a.set_ylim(bottom=0);a.annotate('200: 0.0231',xy=(200,.023136),xytext=(245,.023),arrowprops={'arrowstyle':'->','color':'#444'});a.annotate('400: 0.0111',xy=(400,.011096),xytext=(290,.015))
a=axs[1,0];a.plot(x,[r['q_mean']for r in p],'-^',color='#7B2CBF',markersize=4,label='Q during real evaluation');a.plot(x,[r['actual_mean']for r in p],'-o',color='#009E73',markersize=3,label='Actual discounted return');a.axhspan(0,1,alpha=.12,color='#009E73',label='Valid return range [0, 1]');a.set(ylim=(-1.5,13.5),ylabel='Predicted / actual return',title='C. Q stays wrong even after success recovers');a.legend(fontsize=9,loc='upper left');a.annotate('12.27 vs 0.84',xy=(400,12.275),xytext=(260,12.5))
a=axs[1,1];a.plot(x,[r['training']['lr']*1e5 for r in p],'-D',color='#CC79A7',markersize=4);a.set(ylim=(0,2.8),ylabel='Student learning rate (x 1e-5)',title='D. Student slows down; Q keeps learning');a.text(20,.55,'Student: cosine decay to near zero\nQ: fixed 3e-4; no Q-only warmup',fontsize=10)
fig.suptitle('FQL-fix audit: adjust_bottle | N5 teacher / N1 student | H50 / C50',fontsize=15)
for ext in ['png','svg','pdf']:fig.savefig(O/f'diagnostics.{ext}',dpi=160,bbox_inches='tight')
print(json.dumps({'plots':['diagnostics.png','diagnostics.svg','diagnostics.pdf'],'points':len(p)}))
