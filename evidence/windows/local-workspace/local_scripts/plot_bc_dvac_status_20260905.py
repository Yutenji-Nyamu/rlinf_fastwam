"""Render lightweight, separate learning/optimization/resource views from live JSON."""
import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument('--source', default='docs/rlinf-robotwin-pi0-online-bc/evidence/BC_DVAC_SERVER_REFRESH_20260905_LATEST.json')
p.add_argument('--out', default='docs/rlinf-robotwin-pi0-online-bc/evidence/bc-dvac-status-20260905')
a = p.parse_args()
r = json.loads(Path(a.source).read_text(encoding='utf-8'))
out = Path(a.out); out.mkdir(exist_ok=True)
font = 'C:/Windows/Fonts/msyh.ttc'
orange, teal = '#b9651e', '#117d83'


def series(name, tag, scale=1):
    x = r['runs'][name]['scalars'].get(tag, {})
    points = x.get('all') or x.get('last', [])
    return [(int(i)+1, float(v)*scale) for i, v in points]


def canvas(title, sub, h=940):
    im = Image.new('RGB', (1360, h), '#f6f8fa'); d = ImageDraw.Draw(im)
    def text(x, y, s, size=23, color='#263a4c', anchor=None):
        d.text((x, y), s, fill=color, font=ImageFont.truetype(font, size), anchor=anchor)
    text(55, 25, title, 33)
    text(55, 80, sub, 20, '#5d6c7b')
    return im, d, text


def panel(d, text, top, title, curves, ymax=100):
    left, right, bottom = 110, 1270, top+220
    allpoints = [p for _, _, ps in curves for p in ps]
    xmax = max([p[0] for p in allpoints]+[1])
    text(60, top-40, title, 25)
    d.rounded_rectangle((60, top-5, 1310, bottom+45), radius=10, fill='white')
    def xy(x,y): return left+x/max(1,xmax)* (right-left), bottom-y/ymax*220
    for v in [0, ymax/4, ymax/2, ymax*3/4, ymax]:
        y=xy(0,v)[1]; d.line((left,y,right,y), fill='#dce3e9')
        text(100,y,f'{v:.3g}',17,'#667887','rm')
    ticks=range(0,xmax+1, max(1, xmax//10))
    for x in ticks: text(xy(x,0)[0],bottom+9,str(x),17,'#667887','mt')
    for j,(label,color,points) in enumerate(curves):
        if len(points)>1: d.line([xy(x,y) for x,y in points],fill=color,width=4)
        for x,y in points:
            px,py=xy(x,y); d.ellipse((px-4,py-4,px+4,py+4),fill=color)
        if points: text(1250,top+10+31*j,f'{label}  {points[-1][1]:.4g}',20,color,'rt')


stamp=r['time'][:19].replace('T',' ')
pt=series('sidney','env/success_once',100)
pe=series('sidney','eval/success_once',100)
bt=series('online_bc','env/success_once',100)
be=series('online_bc','eval/success_once',100)
ma=[(x,sum(y for _,y in pt[max(0,i-9):i+1])/len(pt[max(0,i-9):i+1])) for i,(x,_) in enumerate(pt)]
im,d,t=canvas('训练现场：π0.5 GRPO 与 π0 成功在线BC',f'{stamp} 北京时间｜不同任务、不同采样量，分栏看，不作严格方法排名')
panel(d,t,175,'π0.5 · move_pillbottle_pad · 每轮256条／固定评估32条', [('训练MA10 %',orange,ma),('固定评估 %',teal,pe)])
panel(d,t,550,'π0 BC · adjust_bottle · 每轮32条／固定评估32条', [('训练成功率 %',orange,bt),('固定评估 %',teal,be)])
t(60,844,'BC：完成5轮后，第6轮推理CUDA OOM退出；首次固定评估27/32。',23,'#a3402e')
t(60,885,'横轴为完整训练轮次；不补造Step0；BC尚不足以判断持续提升。',20,'#586d7c')
im.save(out/'success.png')

bl=series('online_bc','train/bc/actor_loss')
bg=series('online_bc','train/actor/grad_norm')
im,d,t=canvas('优化指标：BC监督误差与梯度',f'{stamp} 北京时间｜只显示实际完成5轮；下降不等于已证明策略能力提升')
panel(d,t,175,'π0 BC：每轮10次Adam的平均监督FM loss', [('FM loss',orange,bl)],max([y for _,y in bl]+[.001])*1.3)
panel(d,t,550,'π0 BC：梯度范数', [('grad norm',teal,bg)],max([y for _,y in bg]+[.001])*1.3)
t(60,872,'原BC从0.0216降至0.0104；必须结合独立评估判断，不能单看训练loss。',21)
im.save(out/'optimization.png')

gpu=[]
for line in r['gpu']['out'].splitlines():
    f=[v.strip() for v in line.split(',')]
    gpu.append((int(f[0]),float(f[3])/1024,float(f[4])/1024,int(f[5])))
mem={m.group(1):int(m.group(2)) for m in re.finditer(r'^(\w+):\s+(\d+)',r['meminfo'],re.M)}
im,d,t=canvas('服务器资源快照',f'{stamp} 北京时间｜GPU利用率是瞬时采样，不能单点判断卡住',900)
for i,used,total,util in gpu:
    y=150+i*66
    t(65,y,f'GPU {i}',22)
    d.rounded_rectangle((185,y,985,y+33),radius=6,fill='#dfe6eb')
    if used>.02: d.rounded_rectangle((185,y,185+800*used/total,y+33),radius=6,fill=orange if i==6 else teal)
    t(1020,y,f'{used:.1f} GiB · {util}%',21)
t(60,715,f'RAM available：{mem["MemAvailable"]/1024**2:.1f} GiB｜CPU load：{r["loadavg"].split()[0]}/{r["cpu_count"]}',24)
disk=r['disk']['out'].splitlines()[-1].split()
t(60,761,f'/data 可用：{int(disk[3])/1024**3:.1f} GiB（已用{disk[4]}）｜memory / I/O PSI avg10 = 0',23)
t(60,811,'GPU4/5：π0.5；GPU6/7：本轮BC／DVAC smoke用卡，状态另见验收。',21)
t(60,852,'SSH／代理服务active，无failed unit；本轮未做管理员SMART／内核日志审计。',20)
im.save(out/'resources.png')

data={'sidney_train':pt,'sidney_ma10':ma,'sidney_fixed':pe,'bc_train':bt,'bc_fixed':be,'bc_loss':bl,'bc_grad':bg}
html='''<!doctype html><meta charset="utf-8"><title>训练现场</title>
<style>body{max-width:1080px;margin:20px auto;font:16px system-ui;color:#234;background:#f6f8fa}img{width:100%}canvas{width:100%;background:white}select{font:inherit;padding:8px}#tip{padding:10px;min-height:24px}</style>
<h1>训练现场与服务器快照</h1><p>各模型任务不同；已结束BC不代表当前仍在训练。图表从首个真实记录开始；DVAC另做两轮工程smoke，不据此判断学习收益。</p>
<select id="sel"></select><canvas id="c" width="1080" height="400"></canvas><div id="tip">移动鼠标查看数据</div>
<img src="success.png"><details><summary>优化与资源</summary><img src="optimization.png"><img src="resources.png"></details>
<script>const data=DATA;const labels={sidney_train:'π0.5训练成功率 %',sidney_ma10:'π0.5训练MA10 %',sidney_fixed:'π0.5固定评估 %',bc_train:'π0 BC训练成功率 %',bc_fixed:'π0 BC固定评估 %',bc_loss:'BC FM loss',bc_grad:'BC grad norm'};
const sel=document.querySelector('#sel'),c=document.querySelector('#c'),ctx=c.getContext('2d');for(const k in data){let o=new Option(labels[k],k);sel.add(o)}
let ps=[],xm=1,ym=1;function draw(){ps=data[sel.value];xm=Math.max(1,...ps.map(p=>p[0]));ym=Math.max(.001,...ps.map(p=>p[1]))*1.1;ctx.clearRect(0,0,1080,400);ctx.font='18px sans-serif';ctx.fillStyle='#345';ctx.fillText(labels[sel.value],60,30);ctx.strokeStyle='#ddd';for(let i=0;i<=4;i++){let y=350-i*70;ctx.beginPath();ctx.moveTo(70,y);ctx.lineTo(1040,y);ctx.stroke();ctx.fillText((ym*i/4).toPrecision(3),3,y)}ctx.strokeStyle='#117d83';ctx.lineWidth=3;ctx.beginPath();ps.forEach(([x,y],i)=>{let X=70+x/xm*970,Y=350-y/ym*280;if(i)ctx.lineTo(X,Y);else ctx.moveTo(X,Y)});ctx.stroke();for(const [x,y] of ps){ctx.beginPath();ctx.arc(70+x/xm*970,350-y/ym*280,4,0,7);ctx.fill()}ctx.fillText('完整训练轮次（Step）',400,391)}sel.onchange=draw;c.onmousemove=e=>{let x=(e.clientX-c.getBoundingClientRect().left)/c.getBoundingClientRect().width*1080;let p=ps.reduce((a,b)=>Math.abs(70+b[0]/xm*970-x)<Math.abs(70+a[0]/xm*970-x)?b:a,ps[0]);document.querySelector('#tip').textContent=p?`Step ${p[0]}：${p[1].toFixed(5)}`:''};draw();</script>'''.replace('DATA',json.dumps(data))
(out/'index.html').write_text(html,encoding='utf-8')
(out/'metrics.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps({'snapshot':stamp,'output':str(out),'bc_train':bt,'bc_eval':be,'pi05_last':pt[-1],'pi05_ma10':ma[-1],'pi05_fixed':pe[-1],'ram_available_gib':mem['MemAvailable']/1024**2},ensure_ascii=False))
