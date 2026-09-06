"""Build presentation artifacts from the read-only server scalar snapshot."""
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parents[1]
source=root/'docs/rlinf-robotwin-pi0-online-bc/evidence/BC_DVAC_SERVER_REFRESH_20260905_LATEST.json'
r=json.loads(source.read_text(encoding='utf-8'))
s=r['runs']['sidney']['scalars']
train=[(int(x)+1,float(y)*100) for x,y in s['env/success_once']['all']]
ev=[(int(x)+1,float(y)*100) for x,y in s['eval/success_once']['all']]
avg=[sum(y for _,y in train[max(0,i-9):i+1])/len(train[max(0,i-9):i+1]) for i in range(len(train))]
out=root/'docs/rlinf-shenzhen-multitask-pi05/evidence/pi05-bc-review-20260905'
out.mkdir(exist_ok=True)
font='C:/Windows/Fonts/msyh.ttc'
im=Image.new('RGB',(1400,1030),'#f5f7fa');draw=ImageDraw.Draw(im)
def text(x,y,value,size=23,color='#253548',anchor=None):
    draw.text((x,y),value,font=ImageFont.truetype(font,size),fill=color,anchor=anchor)
text(75,27,'π0.5 · Sidney · move_pillbottle_pad',36)
text(75,82,f"现场：{r['time'][:16].replace('T',' ')} 北京时间  |  100 → 200 轮续训",22,'#526071')
left,right=105,1300
xmax=train[-1][0]+5
def point(x,y,top,bottom):return (left+x/xmax*(right-left),bottom-y/100*(bottom-top))
for top,bottom,title in [(190,460,'训练采集：每轮 256 条，深色线为最近10轮均值'),(625,895,'固定评估：32 个固定初态，每5轮评估一次')]:
    text(75,top-55,title,28)
    draw.rounded_rectangle((72,top-8,1330,bottom+12),radius=9,fill='white')
    for tick in [0,25,50,75,100]:
        _,y=point(0,tick,top,bottom);draw.line((left,y,right,y),fill='#dce1e8',width=1)
        text(93,y,f'{tick}%',17,'#667587','rm')
    rx=point(100,0,top,bottom)[0]
    for y in range(top,bottom,13):draw.line((rx,y,rx,min(y+7,bottom)),fill='#8c969f',width=2)
    for x in range(0,train[-1][0]+1,10):text(point(x,0,top,bottom)[0],bottom+23,str(x),17,'#667587','mt')
draw.line([point(x,y,190,460) for x,y in train],fill='#e6b38b',width=2)
draw.line([point(x,avg[i],190,460) for i,(x,_) in enumerate(train)],fill='#ba621e',width=5)
text(1110,201,f'最新 {train[-1][1]:.1f}%',22,'#a45218')
text(950,508,f'最近10轮均值 {avg[-1]:.1f}%',22,'#a45218')
draw.line([point(x,y,625,895) for x,y in ev],fill='#147c83',width=4)
for x,y in ev:
    px,py=point(x,y,625,895);draw.ellipse((px-5,py-5,px+5,py+5),fill='#147c83')
    if x in [5,10,60,70,100,105,110]:text(px,py+(17 if x in [100,110] else -31),f'{round(y*32/100)}/32',20,'#125e64','mt')
text(700,940,'已完成训练轮次（Step）；虚线为 Step100 续训边界',21,'#526071','mt')
text(75,988,'训练采集与固定评估不是同一协议；未记录 Step0，不补造起点。',20,'#526071')
im.save(out/'pi05_progress.png')
with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['completed_step','train_success_256','train_pct','train_ma10_pct','fixed_success_32','fixed_pct'])
    em=dict(ev)
    for i,(x,y) in enumerate(train):w.writerow([x,round(y*256/100),y,avg[i],round(em[x]*32/100) if x in em else '',em.get(x,'')])
html='''<!doctype html><meta charset="utf-8"><title>π0.5 progress</title><style>body{max-width:1100px;margin:24px auto;font:16px system-ui;background:#f5f7fa;color:#234}img{width:100%}table{border-collapse:collapse;width:100%}td,th{text-align:right;padding:7px;border-bottom:1px solid #ddd}</style><h1>π0.5 训练与固定评估</h1><img src="pi05_progress.png"><p>只读现场快照，详细数据见 <a href="metrics.csv">CSV</a>。训练为每轮256条；评估为固定32初态。</p><details><summary>逐次固定评估</summary><table><tr><th>Step</th><th>成功/32</th><th>成功率</th></tr>'''
html+=''.join(f'<tr><td>{x}</td><td>{round(y*32/100)}/32</td><td>{y:.2f}%</td></tr>' for x,y in ev)+'</table></details>'
(out/'index.html').write_text(html,encoding='utf-8')
summary={'snapshot':r['time'],'train_last':train[-1],'train_first10_mean':sum(y for _,y in train[:10])/10,'train_last10_mean':avg[-1],'eval_last':ev[-1],'eval_best':max(ev,key=lambda p:p[1]),'eval_first6_mean':sum(y for _,y in ev[:6])/6,'eval_last6_mean':sum(y for _,y in ev[-6:])/6}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary));print(out/'pi05_progress.png')
