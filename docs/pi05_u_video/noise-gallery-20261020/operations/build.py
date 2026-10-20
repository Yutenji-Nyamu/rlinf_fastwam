import json,html,math,sys,os
from pathlib import Path
sys.path.insert(0,'E:/Codex/home/visualizations/2026/10/06/01a10f72-dbcf-70e1-8236-beac311711d1/plot_deps')
os.environ['MPLCONFIGDIR']='E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a/mpl-cache'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
E=Path(r'E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a')
O=E/'noise4_1020';m=json.loads((O/'manifest.json').read_text());items=m['selected']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
cards=[]
for c in items:
 d=O/c['id'];x=np.array(c['action']);y=np.array(c['noise4']);u=np.array(c['u'])
 selected=[c['options'][i]for i in c['selected_indices']]
 fig,axes=plt.subplots(2,1,figsize=(12,4),sharex=True,layout='constrained')
 for ax,values,color,label in zip(axes,[y,u],['#b55528','#27649b'],['Noise disagreement','U (10 vs 5)']):
  ax.plot(x,values,color=color,lw=1.2);ax.set_ylabel(label);ax.grid(alpha=.15);ax.set_xlim(x[0],x[-1])
  for letter,p in zip('ABC',selected):
   ax.axvline(p['action'],color='#888',lw=.6,ls='--')
   if ax is axes[0]:ax.annotate(letter,(p['action'],values[p['index']]),xytext=(3,7),textcoords='offset points',fontsize=9)
 axes[-1].set_xlabel('Executed action index (controller-projected prefix)')
 fig.suptitle(f"{c['id']} · {c['task']} · seed {c['actual']} · "+('SUCCESS'if c['success']else'FAILURE'),fontsize=12)
 fig.savefig(d/'curve.svg');fig.savefig(d/'curve.png',dpi=110);plt.close(fig)
 pics=''.join(f'<button class="frame" onclick="seek(this,{p["frame"]/c["fps"]:.5f})"><img loading="lazy" src="{c["id"]}/{p["image"]}"><span>{letter} · action {p["action"]} · frame {p["frame"]}</span></button>'for letter,p in zip('ABC',selected))
 note='成功轨迹'if c['success']else'失败轨迹：该任务本次没有成功样本'
 if c['query'][c['peak_index']]==0:
  note+='；最高峰位于首个动作块，画面多为起步/接近阶段，不能仅凭峰值称为接触或错误'
 cards.append(f'''<article data-task="{c['task']}" data-success="{str(c['success']).lower()}"><h2>{c['id']} · {c['task']}</h2><p>{note} · seed {c['actual']} · batch {c['batch_name']} / slot {c['slot']} · {len(x)}个有效动作点</p><img class="curve" loading="lazy" src="{c['id']}/curve.svg"><div class="frames">{pics}</div><details ontoggle="loadVideo(this)"><summary>完整视频、数据与映射依据</summary><video controls preload="none" src="{c['id']}/episode-h264.mp4"></video><p>点击上方帧可定位视频。A/B/C来自峰前／峰值／峰后；每个query选运动较多的手臂进行路径投影。完整曲线保留实际执行前缀，无平滑、无chunk均值；动作与视频帧是投影对应，不是严格一一对应。</p><a href="{c['id']}/series.json">逐动作双信号与选帧数据</a> · <a href="{c['id']}/frame_map.json">完整帧映射</a> · <a href="{c['id']}/curve.svg" download>SVG</a></details></article>''')
for start in range(0,len(items),4):
 group=items[start:start+4];im=Image.new('RGB',(1320,len(group)*340),'white');dr=ImageDraw.Draw(im)
 for row,c in enumerate(group):
  yy=row*340;dr.text((8,yy+4),f"{c['id']} {c['task']} | seed {c['actual']} | {'SUCCESS'if c['success']else'FAILURE'}",font=font,fill='#233d52')
  for j,idx in enumerate(c['selected_indices']):
   p=c['options'][idx];pic=Image.open(O/c['id']/p['image']).convert('RGB');pic.thumbnail((430,280));im.paste(pic,(j*440,yy+30))
   dr.text((j*440+5,yy+310),f"{'ABC'[j]} action {p['action']} / frame {p['frame']} / map err {p['h_error']:.1f}",font=font,fill='#aa3322'if j==1 else'#294859')
 im.save(O/f'contact_{start//4+1:02d}.jpg',quality=94)
missing='、'.join(m['missing_tasks'])
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>四噪声 · 逐动作信号与完整视频</title><style>
body{font:16px/1.65 Arial,"Microsoft Yahei",sans-serif;background:#f3f5f7;color:#26384a;margin:0}main{max-width:1250px;margin:auto;padding:24px}header,article{background:white;padding:24px;border-radius:12px;margin:18px 0}h1{font-size:28px}h2{font-size:21px}.curve{width:100%}.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.frame{border:0;background:#eef2f6;padding:0;cursor:pointer}.frame img{width:100%}.frame span{display:block;padding:8px;font-size:14px}video{max-width:100%;width:780px}input,select{padding:10px;border:1px solid #aaa;border-radius:6px;font-size:16px}a{color:#27649b}summary{cursor:pointer}p{color:#526171}article[hidden]{display:none}
</style><main><header><h1>四种噪声，动作差异有多大？</h1><p>同一观察，4份不同噪声都推理10步；橙色为新信号，蓝色为原U。两者尺度不同，不能凭数值大小判断谁更好。</p>'''
page+=f'<p>1440条有效轨迹中，按峰值/阶段变化清晰度选出{len(items)}条，覆盖{m["tasks"]}个有数据任务，每任务1–2条；成功优先。这是绘图候选，不能用挑选后的案例估计信号准确率。无完整数据：{missing}。</p>'
page+='<input id="q" placeholder="搜索任务或 N 编号" oninput="filter()"><select id="s" onchange="filter()"><option value="">全部</option><option value="true">成功</option><option value="false">失败</option></select><span id="n"></span></header>'+''.join(cards)
page+='''</main><script>function filter(){let q=document.querySelector('#q').value.toLowerCase(),s=document.querySelector('#s').value,n=0;document.querySelectorAll('article').forEach(a=>{let ok=a.textContent.toLowerCase().includes(q)&&(!s||a.dataset.success===s);a.hidden=!ok;n+=ok});document.querySelector('#n').textContent=` 显示 ${n} 条`}function loadVideo(d){let v=d.querySelector('video');if(d.open&&v.preload==='none'){v.preload='metadata';v.load()}}function seek(b,t){let a=b.closest('article'),d=a.querySelector('details'),v=a.querySelector('video');if(v.readyState>=1){v.currentTime=t}else{v.addEventListener('loadedmetadata',()=>{v.currentTime=t},{once:true})}d.open=true;loadVideo(d);v.scrollIntoView({block:'nearest',behavior:'smooth'})}filter()</script></html>'''
(O/'index.html').write_text(page,encoding='utf8')
print(json.dumps({'selected':len(items),'contacts':math.ceil(len(items)/4),'tasks':m['tasks'],'page':str(O/'index.html')}))
