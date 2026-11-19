import json,statistics,math
from pathlib import Path
from datetime import datetime
p=Path(__file__).parent
data=json.loads((p/'capture.json').read_text())
out=[]; scores={}
for g,title in [('5','5 卡 · Dojo DSRL 丢瓶子'),('7','7 卡 · RT2 DSRL-NA 拨开关')]:
 out.append('<article><h2>'+title+'</h2>');scores[g]={}
 for k,label in [('eval/success_once','独立评估'),('env/success_once','训练采样')]:
  a=data[g][k]; assert len(a)>=15 and all(a[i][0]<a[i+1][0] for i in range(len(a)-1))
  series=[a,[(a[i][0],statistics.mean(v for _,v in a[i-4:i+1])) for i in range(4,len(a))]]
  ga=[]
  for i,(r,v) in enumerate(a):
   sub=a[max(0,i-6):i+1];w=[math.exp(-.5*((len(sub)-j-1)/2)**2) for j in range(len(sub))]
   ga.append((r,sum(t[1]*u for t,u in zip(sub,w))/sum(w)))
  series.append(ga)
  n=len(a)//3
  means=[statistics.mean(v for _,v in z) for z in [a[:n],a[n:-n],a[-n:]]]; scores[g][label]=[len(a),a[-1],means]
  x=lambda r:46+(r-1)/(a[-1][0]-1)*471
  y=lambda v:174-v*147
  svg=['<svg viewBox="0 0 536 207">']
  for v in [0,.25,.5,.75,1]:
   svg.append(f'<path d="M46 {y(v)}h471" stroke="#dfE4eb"/><text x="41" y="{y(v)+4}" text-anchor="end">{v:.0%}</text>')
  for t in [1,round(a[-1][0]/2),a[-1][0]]: svg.append(f'<text x="{x(t)}" y="195" text-anchor="middle">{t}</text>')
  for pts,col,width in zip(series,['#97a4b5','#105eaa','#ba5723'],[1,2.6,2]):
   svg.append(f'<polyline stroke="{col}" fill="none" stroke-width="{width}" points="'+' '.join(f'{x(r):.2f},{y(v):.2f}' for r,v in pts)+'"/>')
  for r,v in a:svg.append(f'<circle cx="{x(r):.2f}" cy="{y(v):.2f}" r="2.5" fill="#8995a4"><title>第 {r} 轮 · {round(v*32)}/32</title></circle>')
  out.append('<h3>'+label+'</h3>'+''.join(svg)+'</svg><p>前 / 中 / 后三段均值　'+' → '.join(f'{v:.1%}' for v in means)+'</p>')
 out.append('</article>')
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>5、7 卡成功率趋势</title><style>*{box-sizing:border-box}body{margin:28px auto;max-width:1200px;padding:0 16px;font:16px system-ui;color:#1e3046;background:#f5f7fa}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:16px}article{background:white;padding:15px 22px;border-radius:14px}svg{width:100%;display:block}text{font:12px system-ui;fill:#516073}h3{margin-bottom:0}p{line-height:1.6}h2{font-size:20px}</style><h1>训练和评估成功率趋势</h1><p>横轴为训练轮次。灰线为原始点，<b style="color:#105eaa">蓝线为最近 5 个点均值</b>，<b style="color:#ba5723">橙线为单侧高斯平滑</b>（σ=2 个记录点）。每次训练或评估均为 32 局。三段按记录点数等分，均值使用原始点。</p><section>'''+''.join(out)+'</section></html>'
(p/'trend.html').write_text(page,encoding='utf8');print(json.dumps(scores,ensure_ascii=False))
