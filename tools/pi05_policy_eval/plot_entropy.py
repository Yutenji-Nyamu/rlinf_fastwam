import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V=Path(sys.argv[1])
rows=list(csv.DictReader((V/'student-positions.csv').open()))
def col(k):return np.array([float(r[k]) for r in rows])
x,h,rel,ind=map(col,['distance','student_entropy','relative_dispersion','student_entropy_without_main'])
slots=col('slot').astype(int);succ=col('success') if rows[0]['success'].isdigit() else np.array([r['success']=='True' for r in rows])
def rank(a):
    _,inv,c=np.unique(a,return_inverse=True,return_counts=True);return (np.cumsum(c)-(c+1)/2)[inv]
def rho(a,b):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return float('nan')
    return float(np.corrcoef(rank(a),rank(b))[0,1])
rs={k:rho(x,y) for k,y in [('entropy',h),('relative_dispersion',rel),('entropy_without_main',ind)]}
groups=[np.flatnonzero(slots==s) for s in np.unique(slots)]
rng=np.random.default_rng(1102);boot=[]
for _ in range(500):
    ids=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);boot.append(rho(x[ids],h[ids]))
rs['trajectory_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
per=[dict(slot=int(s),n=int(sum(slots==s)),success=bool(succ[slots==s][0]),rho=rho(x[slots==s],h[slots==s])) for s in np.unique(slots)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
a=ax[0].hexbin(x,h,gridsize=34,mincnt=1,bins='log',cmap='Blues');fig.colorbar(a,ax=ax[0],label='Action count (log)')
ax[0].set(xlabel='Teacher-student action RMSE',ylabel='Student entropy estimate',title=f'All 5,725 positions | Spearman {rs["entropy"]:.3f}')
bins=np.array_split(np.argsort(x,kind='stable'),10)
means=np.array([[x[i].mean(),h[i].mean()] for i in bins])
deciles=[dict(group=j+1,n=len(ids),distance_min=float(x[ids].min()),
    distance_max=float(x[ids].max()),mean_distance=float(x[ids].mean()),
    mean_entropy=float(h[ids].mean())) for j,ids in enumerate(bins)]
with (V/'distance-deciles.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(deciles[0]));w.writeheader();w.writerows(deciles)
ax[1].plot(means[:,0],means[:,1],'-o',color='#086788');ax[1].set(xlabel='Mean action RMSE in distance decile',ylabel='Mean student entropy',title='Same data, ten equal-count groups')
ax[2].bar([r['slot'] for r in per],[r['rho'] for r in per],color=['#179c7d' if r['success'] else '#b7bbc4' for r in per]);ax[2].axhline(0,color='#555',lw=.8);ax[2].set(xlabel='Trajectory slot (green = success)',ylabel='Within-trajectory Spearman correlation',ylim=(-1,1),title='Each trajectory separately')
for ext in ['png','svg','pdf']:fig.savefig(V/f'current-pair.{ext}',dpi=180)
plt.close(fig)
(V/'statistics.json').write_text(json.dumps(dict(positions=len(rows),episodes=32,successes=12,correlations=rs,trajectories=per),indent=2))
ci=rs['trajectory_bootstrap_95']
bin_rows=''.join(f"<tr><td>{r['group']}</td><td>{r['n']}</td><td>{r['mean_distance']:.4f}</td><td>{r['mean_entropy']:.2f}</td></tr>" for r in deciles)
text=f'''<!doctype html><meta charset="utf-8"><title>熵与师生距离 · 采集规划</title><style>body{{max-width:1280px;margin:28px auto;font:16px/1.75 system-ui;color:#203344;padding:0 18px}}img{{width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#087696}}.note{{background:#edf4fa;padding:16px}}</style>
<h1>先看清楚：我们要补哪种数据</h1><p>药瓶 · Clean CP140学生 / Uτ3 CP70教师 · N10/H50/C20 · 32局，12成功；5725个已到达动作位置。</p>
<p>分散度看“同一个策略回答4次差多少”；动作距离看“两个策略各回答1次差多少”。后者已有数据，前者比较两策略时还需教师另外3份。</p>
<h2>已有数据：存在正关联，但不能由此断言高熵就是动作错误</h2><p>当前高斯熵估计与距离的Spearman相关 <b>{rs['entropy']:.3f}</b>；按整条轨迹重抽的95%区间约[{ci[0]:.3f}, {ci[1]:.3f}]。排除参与距离的学生第0份后，剩余3份估熵的相关为 {rs['entropy_without_main']:.3f}。旧相对分散度对应 {rs['relative_dispersion']:.3f}；两个指标定义不同。</p>
<img src="current-pair.png"><p>左图：每点一个动作位置，颜色越深位置越多。中图：仍是同一批数据，按距离分成十组后看平均熵。右图：拆开32条轨迹，避免不同场景混合掩盖局部关系。画法可以让趋势更容易读，不能改数据来强化趋势。</p>
<section id="middle"><h2>中图怎样读：先排队，再分组，最后取平均</h2>
<p><b>一个原始记录是一处动作位置。</b>它有两个数：师生距离＝两个策略在同一观察、同一噪声下的动作差多少；学生熵＝学生用四份不同噪声生成动作时有多分散。</p>
<ol><li>把5725处位置按师生距离从小到大排序。</li><li>切成十组，前五组各573处，后五组各572处。各组人数接近，距离区间宽度可以不同。</li><li>每组画一个点：横坐标是组内平均距离，纵坐标是组内平均学生熵。</li></ol>
<p>从左到右表示师生分歧越来越大的组，<b>不是轨迹时间，也不是训练轮数</b>。点连起来只是帮助看趋势，没有新增数据或拟合平滑曲线。</p>
<p>例如最左组的平均距离为{deciles[0]['mean_distance']:.4f}、平均熵为{deciles[0]['mean_entropy']:.2f}；最右组分别为{deciles[-1]['mean_distance']:.4f}和{deciles[-1]['mean_entropy']:.2f}。负熵来自连续数值的计量尺度，−40高于−50；同一公式下可比较大小。</p>
<p>整体上升表示：分歧较大的组，学生往往也更分散。分组平均会消去一部分位置间的波动，所以线比散点清楚；仍须同时看左图的分散程度。这不代表每一个高熵动作都分歧大，也不代表教师动作一定正确。</p>
<details><summary>展开十组真实数值</summary><table><tr><th>组</th><th>动作位置数</th><th>平均师生距离</th><th>平均学生熵</th></tr>{bin_rows}</table><a href="distance-deciles.csv">十组CSV</a></details>
<p class="note">这张图按“距离”分组，纵轴是“学生熵”。后续“熵变化图”按“初始／学生熵”分组，纵轴是“两模型熵的差值”。两者回答不同问题。</p></section>
<p><a href="current-pair.svg">SVG</a> · <a href="current-pair.pdf">PDF</a> · <a href="student-positions.csv">逐位置CSV</a> · <a href="statistics.json">统计数据</a></p>
<table><tr><th>问题</th><th>新增数据</th><th>图</th></tr><tr><td>高熵时，师生更不一致吗</td><td>药瓶已有；更多轨迹可独立复验</td><td>密度图＋分组均值＋分轨迹</td></tr><tr><td>两个策略在哪里犹豫</td><td>教师补原噪声1/2/3预测</td><td>Top20%位置重合率</td></tr><tr><td>哪些位置变化更多</td><td>与上一项完全共用</td><td>学生熵十组，教师减学生</td></tr><tr><td>训练怎样改变熵</td><td>同run六点＋最终策略固定观察池，当前缺权重</td><td>实际训练步上的熵分位曲线</td></tr></table>
<p class="note">教师补齐改为45卡优先，之后继续原单任务FQL v2，再接RLT。当前两个RL点来自不同run，不能把它们连成训练曲线。完整六点分析需要另一次同run GRPO训练；仅补教师K3不能替代。当前范围仍是药瓶。</p>
<p>公式沿用写作侧K4、14维、无偏方差、1e−6稳定项的对角高斯熵估计。不是模型精确概率熵。实际位置沿用已记录双臂投影的截尾规则。</p>
<p><a href="../review1102c/">服务器与队列</a> · <a href="../single1102c/">完整蒸馏同族</a></p>'''
(V/'index.html').write_text(text,encoding='utf8');print(json.dumps(rs))
