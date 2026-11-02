"""Plot the completed fixed-pool pair; never label two runs as training time."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(directory):
 d=Path(directory)
 summary=json.loads((d/'summary.json').read_text())
 c=summary['paired_comparison']
 rows=list(csv.DictReader((d/'positions.csv').open()))
 hs=np.array([float(r['student_entropy']) for r in rows])
 ht=np.array([float(r['teacher_entropy']) for r in rows])
 assert np.isfinite(hs).all() and np.isfinite(ht).all()
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
 a=ax[0].hexbin(hs,ht,gridsize=35,mincnt=1,bins='log',cmap='Blues')
 bounds=[float(min(hs.min(),ht.min())),float(max(hs.max(),ht.max()))]
 ax[0].plot(bounds,bounds,'--',color='#888',lw=1)
 ax[0].set(xlabel='Student entropy (Clean CP140)',ylabel='Teacher entropy (U tau3 CP70)',title='Same observation and same four noises')
 fig.colorbar(a,ax=ax[0],label='Action count (log)')
 ax[1].bar(['Observed overlap','Random reference'],[100*c['overlap'],100*c['random_reference']],color=['#087f8c','#b8bec6'],width=.5)
 ax[1].set(ylim=(0,100),ylabel='Shared positions / selected positions (%)',title=f'Top 20% positions: {c["top_k"]} per policy')
 for j,v in enumerate([c['overlap'],c['random_reference']]):ax[1].text(j,100*v+2,f'{100*v:.1f}%',ha='center')
 delta=np.array([r['delta'] for r in c['bins']])
 ax[2].bar(np.arange(1,11),delta,color=['#087f8c' if x>=0 else '#d97736' for x in delta])
 ax[2].axhline(0,color='#555',lw=.8)
 ax[2].set(xticks=range(1,11),xlabel='Groups defined by student entropy (low to high)',ylabel='Mean teacher entropy minus student entropy',title='Keep the same positions in each group')
 for ext in ['png','svg','pdf']:fig.savefig(d/f'paired-entropy.{ext}',dpi=180)
 plt.close(fig)
 with (d/'entropy-deciles.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(c['bins'][0]));w.writeheader();w.writerows(c['bins'])
 html=f'''<!doctype html><meta charset="utf-8"><title>药瓶师生双熵 · 补算结果</title>
<style>body{{max-width:1280px;margin:28px auto;font:16px/1.75 system-ui;color:#203344;padding:0 18px}}img{{width:100%}}a{{color:#087696}}.note{{background:#edf4fa;padding:16px}}</style>
<h1>同一批位置，两个策略各推理四次</h1>
<p>Clean CP140学生实际执行，Uτ3 CP70教师在相同观察、相同四份噪声下旁算。{summary['episodes']}局、{summary['positions']}个已到达动作位置；原视频和学生预测沿用，不是新收轨迹。</p>
<img src="paired-entropy.png">
<p><b>左图：</b>每处位置各有学生熵和教师熵。虚线上两者一样大；线下教师更集中，线上教师更分散。</p>
<p><b>中图：</b>两策略各选熵最高的20%位置，各{c['top_k']}处，其中{100*c['overlap']:.1f}%重合。独立随机选取的期望约20%。这衡量“犹豫的位置是否相同”，不是成功率。</p>
<p><b>右图：</b>先按学生熵从低到高分十组，始终跟踪原来这些位置。每柱是组内平均的“教师熵−学生熵”；正值表示教师更分散，负值表示教师更集中。它与按师生距离分组、画平均学生熵的旧中图不同。</p>
<p class="note">这两个权重来自不同训练run，因此本页是策略配对比较，不能解释成同一策略从早到晚的训练变化。只有完整GRPO同run六点才有训练时间横轴。K4对角高斯熵是动作分散度估计，不是动作正确性。</p>
<p><a href="paired-entropy.svg">SVG</a> · <a href="paired-entropy.pdf">PDF</a> · <a href="positions.csv">逐位置双熵与距离CSV</a> · <a href="entropy-deciles.csv">十组CSV</a> · <a href="summary.json">汇总JSON</a></p>
<p><a href="../entropy1102/#middle">原中图的详细读法</a></p>'''
 (d/'index.html').write_text(html,encoding='utf8')
 print(json.dumps({'positions':len(rows),'overlap':c['overlap'],'delta_by_group':delta.tolist()}))

if __name__=='__main__':main(sys.argv[1])
