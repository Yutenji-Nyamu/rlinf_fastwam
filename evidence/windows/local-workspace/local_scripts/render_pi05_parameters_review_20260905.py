"""Read-only snapshot -> static/mobile plots and a self-contained hover dashboard."""
import csv, datetime, html, json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

base=Path(__file__).resolve().parents[1]
ev=base/'docs/rlinf-robotwin-pi0-online-bc/evidence'
r=json.loads((ev/'BC_PARAMETERS_STORAGE_LIVE_20260905.json').read_text(encoding='utf-8'))
f=json.loads((ev/'BC_PARAMETERS_FOLLOWUP_LIVE_20260905.json').read_text(encoding='utf-8'))
out=base/'docs/rlinf-shenzhen-multitask-pi05/evidence/pi05-bc-parameters-20260905'
out.mkdir(exist_ok=True)
s=r['sidney']['scalars'];s.update(f['scalars'])
def series(tag,scale=1):
    by_step={p['step']:p for p in sorted(s[tag],key=lambda p:p['wall_time'])}
    return [[x,p['value']*scale] for x,p in sorted(by_step.items())]
train=series('env/success_once',100);fixed=series('eval/success_once',100)
ma=[[x,sum(z[1] for z in train[max(0,i-9):i+1])/min(i+1,10)] for i,(x,y) in enumerate(train)]
xmax=math.ceil(train[-1][0]/10)*10
resource=[row for v in r['sidney']['resource'].values() for row in v['rows']]
resource.sort(key=lambda row:row['timestamp'])
t0=datetime.datetime.fromisoformat(resource[0]['timestamp']).timestamp()
def resources(key,scale):
    return [[(datetime.datetime.fromisoformat(p['timestamp']).timestamp()-t0)/3600,float(p[key])*scale]
            for p in resource if (p.get(key) or '').replace('.','',1).isdigit()]
panels={
 'success':[
  {'title':'训练采集：每轮256条','xmax':xmax,'ymax':100,'xlabel':'已完成训练轮次','ylabel':'成功率 %','guide':100,
   'lines':[('逐轮','#e3b693',train),('最近10轮均值','#b85d1e',ma)]},
  {'title':'固定评估：每5轮，32个固定初态','xmax':xmax,'ymax':100,'xlabel':'已完成训练轮次','ylabel':'成功率 %','guide':100,
   'lines':[('固定32','#087e84',fixed)]}],
 'optimization':[
  {'title':'梯度范数（裁剪前记录）','xmax':xmax,'xlabel':'已完成训练轮次','ylabel':'grad norm','guide':100,
   'lines':[('train/actor/grad_norm','#b85d1e',series('train/actor/grad_norm'))]},
  {'title':'近似KL与裁剪比例（不等于成功率）','xmax':xmax,'xlabel':'已完成训练轮次','ylabel':'数值','guide':100,
   'lines':[('approx_kl','#087e84',series('train/actor/approx_kl')),('clip_fraction','#b85d1e',series('train/actor/clip_fraction'))]}],
 'resources':[
  {'title':'GPU4/5显存（资源曲线抽样，非精确峰值）','xlabel':'首次启动后小时','ylabel':'GiB','ymax':80,
   'lines':[('GPU4','#b85d1e',resources('gpu4_used_mib',1/1024)),('GPU5','#087e84',resources('gpu5_used_mib',1/1024))]},
  {'title':'全机可用内存（含其他任务影响）','xlabel':'首次启动后小时','ylabel':'GiB',
   'lines':[('MemAvailable','#087e84',resources('host_mem_available_kib',1/1024**2))]}]}
font='C:/Windows/Fonts/msyh.ttc'
def txt(d,x,y,t,size=21,fill='#283e50',anchor=None):
    d.text((x,y),t,font=ImageFont.truetype(font,size),fill=fill,anchor=anchor)
svgs={}
for name,ps in panels.items():
    im=Image.new('RGB',(1380,1000),'#f4f7fa');d=ImageDraw.Draw(im)
    title={'success':'π0.5 GRPO · 训练与固定评估','optimization':'π0.5 GRPO · 优化指标','resources':'π0.5 GRPO · 资源记录'}[name]
    txt(d,60,20,title,35);txt(d,60,70,f"move_pillbottle_pad  |  指标 {f['time'][:16].replace('T',' ')} CST",21)
    svg=[f'<svg viewBox="0 0 1380 1000" role="img" aria-label="{title}"><rect width="1380" height="1000" fill="#f4f7fa"/><text x="60" y="50" font-size="32">{title}</text><text x="60" y="92" font-size="20">{f["time"][:16]} CST</text>']
    for idx,p in enumerate(ps):
        left,right,top,bottom=110,1280,170+idx*405,435+idx*405
        xx=p.get('xmax',math.ceil(max(x for _,_,values in p['lines'] for x,y in values)))
        vals=[y for _,_,a in p['lines'] for x,y in a if math.isfinite(y)]
        ymin=min(0,min(vals)); ymax=p.get('ymax',max(vals)*1.1 or 1)
        def pt(x,y):return left+x/xx*(right-left),bottom-(y-ymin)/(ymax-ymin)*(bottom-top)
        txt(d,60,top-52,p['title'],26)
        d.rounded_rectangle((65,top-10,1320,bottom+15),10,fill='white')
        svg.append(f'<text x="60" y="{top-25}" font-size="26">{p["title"]}</text><rect x="65" y="{top-10}" width="1255" height="{bottom-top+25}" rx="10" fill="white"/>')
        for j in range(5):
            y=ymin+j*(ymax-ymin)/4; py=pt(0,y)[1];lab=f'{y:.3g}'
            d.line((left,py,right,py),fill='#dce3e8');txt(d,100,py,lab,17,anchor='rm')
            svg.append(f'<path d="M {left} {py} H {right}" stroke="#dce3e8"/><text x="100" y="{py+6}" text-anchor="end" font-size="17">{lab}</text>')
        for j in range(7):
            x=j*xx/6;px=pt(x,0)[0];lab=f'{x:.0f}'
            txt(d,px,bottom+25,lab,17,anchor='mt')
            svg.append(f'<text x="{px}" y="{bottom+42}" text-anchor="middle" font-size="17">{lab}</text>')
        if p.get('guide'):
            px=pt(p['guide'],0)[0]
            for yy in range(top,bottom,12):d.line((px,yy,px,min(yy+6,bottom)),fill='#8d99a3',width=2)
            svg.append(f'<path d="M{px},{top} V{bottom}" stroke="#8d99a3" stroke-dasharray="6 6"/>')
        for k,(label,color,values) in enumerate(p['lines']):
            points=[pt(x,y) for x,y in values if math.isfinite(y)]
            if len(points)>1:d.line(points,fill=color,width=3 if k==0 else 5)
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="'+ ' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'"/>')
            for x,y in values:
                if not math.isfinite(y):continue
                px,py=pt(x,y)
                svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5" fill="{color}" fill-opacity="0.1"><title>{html.escape(label)} | x={x:.2f} | y={y:.5g}</title></circle>')
                if name=='success' and idx==1:
                    d.ellipse((px-4,py-4,px+4,py+4),fill=color)
                    if x in (5,10,70,100,125):txt(d,px,py-28,f'{round(y*.32)}/32',18,color,'mt')
            txt(d,110+k*500,bottom+62,f'{label}：最新 {values[-1][1]:.4g}',20,color)
            svg.append(f'<text x="{110+k*500}" y="{bottom+82}" fill="{color}" font-size="20">{html.escape(label)}：最新 {values[-1][1]:.4g}</text>')
    foot='虚线为Step100续训边界；未记录Step0，不补造起点。' if name!='resources' else '资源截取至22:56；全机内存不能全部归因于该实验。'
    txt(d,60,964,foot,19);svg.append(f'<text x="60" y="985" font-size="19">{foot}</text></svg>')
    im.save(out/f'{name}.png');svgs[name]=''.join(svg)
em=dict(fixed)
with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as h:
    w=csv.writer(h);w.writerow(['completed_step','train_success_256','train_pct','train_ma10_pct','fixed_success_32'])
    for (x,y),(_,avg) in zip(train,ma):w.writerow([x,round(y*2.56),y,avg,round(em[x]*.32) if x in em else ''])
page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>π0.5 GRPO现场</title><style>body{max-width:1100px;margin:20px auto;padding:0 12px;background:#f4f7fa;font:17px system-ui;color:#283e50}button{padding:10px 20px;margin:4px;cursor:pointer}svg{width:100%;font-family:system-ui,"Microsoft YaHei"}circle:hover{fill-opacity:1}section{display:none}section.active{display:block}</style><h1>π0.5 GRPO只读现场</h1><p>点击切换；悬停曲线点查看数值。<a href="metrics.csv">逐轮CSV</a>。</p>'''
for n,t in [('success','成功率'),('optimization','优化指标'),('resources','资源')]:page+=f'<button onclick="show(\'{n}\')">{t}</button>'
for n,svg in svgs.items():page+=f'<section id="{n}" class="{("active" if n=="success" else "")}">{svg}</section>'
page+='<script>function show(id){document.querySelectorAll("section").forEach(x=>x.classList.toggle("active",x.id===id))}</script>'
(out/'index.html').write_text(page,encoding='utf-8')
summary={'time':f['time'],'complete':int(f['log_step'][-1]),'train_last':train[-1],'train_ma10':ma[-1][1],
         'fixed_last':fixed[-1],'fixed_best':max(fixed,key=lambda p:p[1]),'fixed_recent':fixed[-6:],
         'train_first10_mean':sum(y for _,y in train[:10])/10,'mean_recent_step_seconds':sum(y for _,y in series('time/step')[-10:])/10}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False));print(out)
