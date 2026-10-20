import json,sys,os,html,math,collections
from pathlib import Path
sys.path.insert(0,'E:/Codex/home/visualizations/2026/10/06/01a10f72-dbcf-70e1-8236-beac311711d1/plot_deps')
E=Path('E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a')
O=E/'dojo_u_1020';os.environ['MPLCONFIGDIR']=str(E/'mpl-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
m=json.loads((O/'manifest.json').read_text());items=m['selected']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
cards=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for c in items:
 d=O/c['id'];fig,ax=plt.subplots(figsize=(13,3.2),layout='constrained')
 ax.plot(c['action'],c['u'],color='#27649b',lw=1.25);ax.grid(alpha=.18)
 ax.set(xlim=(0,c['action'][-1]),xlabel='Executed action index',ylabel='U (10 vs 5)')
 for f in c['selected_frames']:
  ax.axvline(f['action'],color='#777',ls='--',lw=.7)
  ax.annotate(f['label'],(f['action'],f['u']),xytext=(4,7),textcoords='offset points',fontsize=12)
 ax.set_title(f"{c['id']} | {c['task']} | episode {c['episode_id']} | "+('SUCCESS'if c['success']else'FAILURE'))
 fig.savefig(d/'curve.svg');fig.savefig(d/'curve.png',dpi=110);plt.close(fig)
 pics='';wrists=''
 for f in c['selected_frames']:
  p=f['cameras']['cam_head']
  pics+=f'<button class="frame" onclick="seek(this,{p["frame"]/c["fps"]:.5f})"><img loading="lazy" src="{c["id"]}/{p["image"]}"><span>{f["label"]} · action {f["action"]} · frame {p["frame"]}</span></button>'
 for cam in ['cam_left_wrist','cam_right_wrist']:
  wrists+='<p>'+cam+'</p><div class="frames">'+''.join(f'<div class="frame"><img loading="lazy" src="{c["id"]}/{f["cameras"][cam]["image"]}"><span>{f["label"]} · action {f["action"]}</span></div>'for f in c['selected_frames'])+'</div>'
 note='成功轨迹'if c['success']else'失败轨迹：该任务当前没有成功样本'
 src=c['source_layout'];note+=f' · episode {c["episode_id"]} · layout seed {src["layout_seed"]}/{src["source_layout_id"]} · repeat {src["repeat"]}'
 if c['peak_index']%50 in (0,1,48,49):note+='；选中峰靠近动作块边界，需留意重新观察造成的变化'
 cards.append(f'<article data-task="{c["task"]}" data-success="{str(c["success"]).lower()}"><h2>{c["id"]} · {c["task"]}</h2><p>{note} · {len(c["u"])}动作</p><img class="curve" src="{c["id"]}/curve.svg"><div class="frames">{pics}</div><details><summary>同一时刻的左右腕部画面</summary>{wrists}</details><details ontoggle="loadVideo(this)"><summary>完整主相机视频、数据与SVG</summary><video controls preload="none" src="{c["id"]}/episode.mp4"></video><p>点击A/B/C定位。按原生action → query/h → control tick → 实际视频帧匹配；曲线完整、无平滑，不画未执行的预测尾部。</p><a href="{c["id"]}/series.json">逐动作U</a> · <a href="{c["id"]}/frames.json">三相机帧映射</a> · <a href="{c["id"]}/curve.svg">SVG</a></details></article>')
for start in range(0,len(items),2):
 group=items[start:start+2];im=Image.new('RGB',(990,570*len(group)),'white');dr=ImageDraw.Draw(im)
 for row,c in enumerate(group):
  y0=row*570;dr.text((4,y0+3),f'{c["id"]} {c["task"]} | episode {c["episode_id"]} | success={c["success"]}',font=font,fill='#25384a')
  for j,f in enumerate(c['selected_frames']):
   for k,cam in enumerate(['cam_head','cam_left_wrist','cam_right_wrist']):
    pic=Image.open(O/c['id']/f['cameras'][cam]['image']).convert('RGB');pic.thumbnail((320,160))
    im.paste(pic,(j*330,y0+30+k*178))
    dr.text((j*330+2,y0+190+k*178),f'{f["label"]} a{f["action"]} {cam}',font=font,fill='#36546a')
 im.save(O/f'contact_{start//2+1:02d}.jpg',quality=93)
css=(E/'noise4_1020/index.html').read_text(encoding='utf8').split('<style>')[1].split('</style>')[0]
js=(E/'noise4_1020/index.html').read_text(encoding='utf8').split('<script>')[1].split('</script>')[0]
total=sum(x['episodes']for x in m['tasks'].values());success=sum(x['successes']for x in m['tasks'].values())
table='<table><tr><th>任务</th><th>已完成回合</th><th>成功</th><th>成功率</th><th>每轨迹U中位数的中位数</th></tr>'
for task,s in m['tasks'].items():
 table+=f'<tr><td>{task}</td><td>{s["episodes"]}</td><td>{s["successes"]}</td><td>{100*s["successes"]/s["episodes"]:.1f}%</td><td>{s["episode_median_u"]:.4f}</td></tr>'
table+='</table>'
page=f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dojo U · 完整轨迹与真实帧</title><style>{css}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style><main><header><h1>Dojo π0.5：U与完整执行轨迹</h1><p>深圳3正式采集快照：{total}条已封装回合，{success}条成功，10个任务有回合。另有深圳1 smoke 19/25成功，未混入此表。采集仍在继续。</p><p>同一观察、同一噪声，完整推理10步与5步动作的逐位置分歧。成功任务各选2条，零成功任务各1条失败案例，共{len(items)}条；这是画图筛选，不能用精选例子判断U预测失败的准确率。动作块边界可能出现跳变；高U也可能出现在成功操作中。</p>{table}<p>尚无完整回合的任务不伪造曲线；完整失败/跳过记录保留在manifest.json。布局可能重复，回合数不等于独立布局数。</p><input id="q" placeholder="搜索任务或 D 编号" oninput="filter()"><select id="s" onchange="filter()"><option value="">全部</option><option value="true">成功</option><option value="false">失败</option></select><span id="n"></span></header>'+''.join(cards)+f'</main><script>{js}</script></html>'
(O/'index.html').write_text(page,encoding='utf8')
(O/'coverage.json').write_text(json.dumps({'total':total,'successes':success,'tasks':m['tasks'],'selected':len(items)},indent=2))
print(json.dumps({'episodes':total,'successes':success,'selected':len(items),'contacts':math.ceil(len(items)/2)}))
