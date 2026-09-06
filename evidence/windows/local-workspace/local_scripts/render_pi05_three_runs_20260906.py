"""Render a saved read-only server snapshot. No SSH, model execution, or training."""
import argparse, csv, datetime, html, json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'docs/rlinf-robotwin-pi0-online-bc/evidence'
parser=argparse.ArgumentParser()
parser.add_argument('--input', default='PI05_THREE_RUNS_READONLY_REFRESH_20260906.json')
parser.add_argument('--output', default='pi05-three-runs-20260906')
args=parser.parse_args()
raw=json.loads((EV/args.input).read_text(encoding='utf-8'))
out=EV/args.output;out.mkdir(exist_ok=True)
ORANGE='#bc6019';TEAL='#007b80';GRAY='#516a83';LIGHT='#9fb4c4'
LABEL={'bc':'在线BC','dvac':'BC + DVAC','grpo':'GRPO'}
COLOR={'bc':ORANGE,'dvac':TEAL,'grpo':GRAY}
FONT='C:/Windows/Fonts/msyh.ttc'
tz=datetime.timezone(datetime.timedelta(hours=8))
def arr(name,tag,scale=1):
    return [(p['step'],p['value']*scale) for p in raw['runs'][name]['scalars'].get(tag,[]) if math.isfinite(p['value'])]
def avg(a):return sum(a)/len(a) if a else None
def moving(a,n=10):return [(x,avg([v for _,v in a[max(0,i-n+1):i+1]])) for i,(x,y) in enumerate(a)]
def line(name,tag,scale=1,ma=False):
    data=arr(name,tag,scale)
    return (LABEL[name],COLOR[name],moving(data) if ma else data)
def rows(name):
    return sorted({p['timestamp']:p for r in raw['runs'][name]['resource'] for p in r['sample']}.values(),key=lambda p:p['timestamp'])
t0=datetime.datetime.fromisoformat(raw['runs']['bc']['state']['started_at.txt']).timestamp()
def resource(name,key,scale=1):
    result=[]
    for row in rows(name):
        try:
            x=(datetime.datetime.fromisoformat(row['timestamp']).timestamp()-t0)/3600;y=float(row[key])*scale
            if x>=0 and math.isfinite(y):result.append((x,y))
        except (ValueError,KeyError,TypeError):pass
    return result
summary={'time':raw['time'],'finished_read_at':raw['finished_read_at'],'runs':{}}
for name,r in raw['runs'].items():
    train=arr(name,'env/success_once',100);ev=arr(name,'eval/success_once',100)
    scalar=r['scalars'];last=scalar['time/step'][-1];step=int(r['step_mentions'][-1][0]);ck=r['checkpoints'][-1]
    peak={key:max(res['max'].get(key,0) for res in r['resource']) for key in r['resource'][0]['max']}
    summary['runs'][name]={'completed':step,'target':int(r['step_mentions'][-1][1]),'wrapper_alive':r['wrapper']['alive'],
        'latest_train_pct':train[-1][1],'latest_train_success':round(train[-1][1]/100*r['attempts_per_round']),
        'train_ma10_pct':avg([v for _,v in train[-10:]]),'train_first10_pct':avg([v for _,v in train[:10]]),
        'fixed_latest':ev[-1],'fixed_recent':ev[-6:],'fixed_best':max(ev,key=lambda p:p[1]),
        'fixed_first4_mean_pct':avg([v for _,v in ev[:4]]),'fixed_last4_mean_pct':avg([v for _,v in ev[-4:]]),
        'checkpoint_step':ck['step'],'checkpoint_gib':ck['bytes']/1024**3,'checkpoint_total_gib':sum(c['bytes'] for c in r['checkpoints'])/1024**3,
        'error_hits':r['error_hit_count'],'attempts_completed':step*r['attempts_per_round'],
        'recent10_mean_seconds':{tag:avg([v for _,v in arr(name,tag)[-10:]]) for tag in ['time/step','time/generate_rollouts','time/actor_training','time/sync_weights','time/eval']},
        'resource_peaks':peak,
        'eta_if_recent_pace_holds':datetime.datetime.fromtimestamp(last['wall_time']+(int(r['step_mentions'][-1][1])-step)*avg([v for _,v in arr(name,'time/step')[-10:]]),tz).isoformat(),
        'nonfinite_scalar_count':sum(not math.isfinite(p['value']) for points in scalar.values() for p in points)}
    if name!='grpo':
        summary['runs'][name]['fm_loss_first_last']=[arr(name,'train/bc/actor_loss')[0][1],arr(name,'train/bc/actor_loss')[-1][1]]
        summary['runs'][name]['pool_queries']=arr(name,'train/replay_buffer/query_records')[-1][1]
        summary['runs'][name]['success_episodes']=arr(name,'train/replay_buffer/success_episodes')[-1][1]
        summary['runs'][name]['train_step31_40_mean_pct']=avg([v for x,v in train if 31<=x<=40])
        summary['runs'][name]['first_collection_success']=round(train[0][1]*.32)
    if name=='dvac':
        summary['runs'][name]['weight_latest']={key:arr(name,'train/dvac/weight_'+key)[-1][1] for key in ['min','mean','max','std']}

panels={
 'success':[
 {'title':'BC对照 · 固定32初态评估（每5轮）','xlabel':'完成轮次（两项均32条训练尝试/轮）','ymax':100,'xmax':45,'count_labels':True,
  'lines':[line(n,'eval/success_once',100) for n in ['bc','dvac']]},
 {'title':'GRPO · 固定32初态评估（每5轮）','xlabel':'完成轮次（256条训练尝试/轮）','ymax':100,'xmax':160,'guide':100,
  'lines':[line('grpo','eval/success_once',100)]}],
 'training':[
 {'title':'BC采集成功率 · 细线逐轮，粗线最近10轮均值','xlabel':'采集轮次（第1轮在首次更新前：BC 16/32，DVAC 9/32）','ymax':100,'xmax':45,'raw_lines':2,
  'lines':[(LABEL[n]+'逐轮',c,arr(n,'env/success_once',100)) for n,c in [('bc','#dfb99a'),('dvac','#97c8c7')]]+[line(n,'env/success_once',100,True) for n in ['bc','dvac']]},
 {'title':'GRPO采集成功率 · 最近10轮均值','xlabel':'完成轮次（256条/轮）','ymax':100,'xmax':160,'guide':100,
  'lines':[line('grpo','env/success_once',100,True)]}],
 'optimization':[
 {'title':'BC原生FM loss（DVAC为加权loss，数值不能直接比优劣）','xlabel':'完成轮次','xmax':45,
  'lines':[line(n,'train/bc/actor_loss') for n in ['bc','dvac']]},
 {'title':'DVAC实际逐动作权重 · 均值1，未越出[0.5, 1.5]','xlabel':'完成轮次','xmax':45,'ymax':1.5,
  'lines':[(k,c,arr('dvac','train/dvac/weight_'+tag)) for k,c,tag in [('最小',ORANGE,'min'),('均值',GRAY,'mean'),('最大',TEAL,'max')]]},
 {'title':'GRPO裁剪前梯度范数（与BC不是同一损失尺度）','xlabel':'完成轮次','xmax':160,'guide':100,
  'lines':[line('grpo','train/actor/grad_norm')]}],
 'resources':[
 {'title':'GPU显存 · 采样曲线，80GB卡实际总量约79.65GiB','xlabel':'BC启动后小时（09-05 23:38起）','ymax':80,
  'lines':[(f'GPU{i}',c,resource(n,f'gpu{i}_used_mib',1/1024)) for n,i,c in [('grpo',4,GRAY),('grpo',5,LIGHT),('bc',6,ORANGE),('dvac',7,TEAL)]]},
 {'title':'整机可用RAM · 下降趋势需要关注（不是显存）','xlabel':'BC启动后小时','ylabel':'GiB',
  'lines':[('MemAvailable',GRAY,resource('bc','host_mem_available_kib',1/1024**2))]},
 {'title':'BC环境进程RSS · 增长尚未解释，不能直接等同泄漏','xlabel':'BC启动后小时','ylabel':'GiB',
        'lines':[(LABEL[n],COLOR[n],resource(n,'env_rss_kib',1/1024**2)) for n in ['bc','dvac']]}]}

for p in panels['success']:p['is_success']=True
panels['bc_comparison']=[panels['training'][0],panels['success'][0]]
panels['overview']=[panels['training'][0],panels['success'][0],panels['training'][1],panels['success'][1]]
snapshot_time=datetime.datetime.fromisoformat(raw['time']).strftime('%Y-%m-%d %H:%M CST')
subtitle=f"服务器只读快照：{snapshot_time}  |  BC {summary['runs']['bc']['completed']}轮 · DVAC {summary['runs']['dvac']['completed']}轮 · GRPO {summary['runs']['grpo']['completed']}轮"

def txt(d,x,y,text,size=20,fill='#283e50',anchor=None):
    d.text((x,y),text,font=ImageFont.truetype(FONT,size),fill=fill,anchor=anchor)
svgs={}
for name,ps in panels.items():
    height=170+410*len(ps);im=Image.new('RGB',(1380,height),'#f4f7fa');d=ImageDraw.Draw(im)
    title={'success':'π0.5 · 固定评估','training':'π0.5 · 训练采集','optimization':'π0.5 · 优化与DVAC信号','resources':'π0.5 · 服务器资源','bc_comparison':'π0.5 BC · 训练采集 + 固定评估','overview':'π0.5 · 训练采集 + 固定评估'}[name]
    txt(d,55,20,title,34);txt(d,55,70,subtitle,21)
    svg=[f'<svg viewBox="0 0 1380 {height}" role="img" aria-label="{title}"><rect width="1380" height="{height}" fill="#f4f7fa"/><text x="55" y="50" font-size="34">{title}</text><text x="55" y="94" font-size="21">{subtitle}</text>']
    for index,p in enumerate(ps):
        left,right,top,bottom=105,1290,165+410*index,415+410*index
        vals=[v for _,_,points in p['lines'] for _,v in points];xx=p.get('xmax',math.ceil(max(x for _,_,points in p['lines'] for x,_ in points)))
        lo=min(0,min(vals));hi=p.get('ymax',max(vals)*1.12 or 1)
        def pos(x,y):return left+x/xx*(right-left),bottom-(y-lo)/(hi-lo)*(bottom-top)
        txt(d,55,top-45,p['title'],25)
        d.rounded_rectangle((60,top-8,1325,bottom+12),10,fill='white')
        svg.append(f'<text x="55" y="{top-19}" font-size="25">{html.escape(p["title"])}</text><rect x="60" y="{top-8}" width="1265" height="{bottom-top+20}" rx="10" fill="white"/>')
        for j in range(5):
            y=lo+j*(hi-lo)/4;py=pos(0,y)[1];lab=f'{y:.3g}'
            d.line((left,py,right,py),fill='#dce3e8');txt(d,95,py,lab,17,anchor='rm')
            svg.append(f'<path d="M{left},{py} H{right}" stroke="#dce3e8"/><text x="95" y="{py+6}" text-anchor="end" font-size="17">{lab}</text>')
        tick_step=5 if xx==45 else 20 if xx==160 else 2
        for x in range(0,int(xx)+1,tick_step):
            px=pos(x,0)[0];lab=f'{x:.0f}'
            txt(d,px,bottom+20,lab,17,anchor='mt');svg.append(f'<text x="{px}" y="{bottom+38}" text-anchor="middle" font-size="17">{lab}</text>')
        if p.get('guide'):
            px=pos(p['guide'],0)[0]
            for yy in range(top,bottom,12):d.line((px,yy,px,min(yy+6,bottom)),fill='#8d99a3',width=2)
            svg.append(f'<path d="M{px},{top} V{bottom}" stroke="#8d99a3" stroke-dasharray="6 6"/>')
        for k,(label,color,points) in enumerate(p['lines']):
            coords=[pos(x,y) for x,y in points]
            width=2 if k<p.get('raw_lines',0) else 4
            if len(coords)>1:d.line(coords,fill=color,width=width)
            svg.append('<polyline fill="none" stroke="'+color+'" stroke-width="'+str(width)+'" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)+'"/>')
            for x,y in points:
                px,py=pos(x,y)
                if p.get('is_success'):
                    if k==1:d.rectangle((px-5,py-5,px+5,py+5),fill=color)
                    else:d.ellipse((px-5,py-5,px+5,py+5),fill=color)
                    if p.get('count_labels'):
                        other=dict(p['lines'][1-k][2]).get(x,y)
                        dy=-30 if y>other or (y==other and k==0) else 15
                        txt(d,px,py+dy,f'{round(y*.32)}',18,color,'mt')
                svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6" fill="{color}" fill-opacity=".15"><title>{html.escape(label)} | x={x:.2f} | y={y:.6g}</title></circle>')
            if k<p.get('raw_lines',0):continue
            value=f'{round(points[-1][1]*.32)}/32' if p.get('is_success') else f'{points[-1][1]:.4g}'
            if p.get('raw_lines'):value+=' %（MA10）'
            legend=f'{label}：最新 {value}'
            effective_count=len(p['lines'])-p.get('raw_lines',0)
            lx=105+(k-p.get('raw_lines',0))*(295 if effective_count>2 else 505)
            txt(d,lx,bottom+55,legend,20,color);svg.append(f'<text x="{lx}" y="{bottom+79}" font-size="20" fill="{color}">{html.escape(legend)}</text>')
        txt(d,690,bottom+99,p['xlabel'],18,anchor='mt')
        svg.append(f'<text x="690" y="{bottom+120}" font-size="18" text-anchor="middle">{html.escape(p["xlabel"])}</text>')
    foot='成功率单位%；点旁数字为成功条数/32。首轮采集不等于Step0固定评估；无新增推理。' if name in ['success','bc_comparison','overview'] else '每轮交互量不同，不能按相同步数比较效率；GRPO虚线为Step100续训。' if name in ['training','optimization'] else '显存/RAM/RSS单位GiB；采样峰值见报告。RSS含共享映射，不能直接相加当私有占用。'
    txt(d,55,height-30,foot,18);svg.append(f'<text x="55" y="{height-9}" font-size="18">{foot}</text></svg>')
    im.save(out/f'{name}.png');svgs[name]=''.join(svg)

with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.writer(f);writer.writerow(['run','tag','completed_step','value','wall_time'])
    for name,r in raw['runs'].items():
        for tag,points in r['scalars'].items():
            for p in points:writer.writerow([name,tag,p['step'],p['value'],p['wall_time']])
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>π0.5 三项实验现场 2026-09-06</title><style>body{max-width:1200px;margin:20px auto;padding:0 12px;background:#f4f7fa;font:17px system-ui;color:#283e50}button{padding:10px 20px;margin:4px;cursor:pointer}svg{width:100%;font-family:system-ui,"Microsoft YaHei"}circle:hover{fill-opacity:1}section{display:none}section.active{display:block}</style><h1>π0.5 三项实验 · 只读现场</h1><p>10:01 CST；三项继续运行，无所查fatal/OOM。固定评估尚不能证明DVAC稳定优于BC。注意环境RAM增长。点击切换，悬停查值。<a href="metrics.csv">全部指标CSV</a> | <a href="../PI05_THREE_RUNS_READONLY_REFRESH_20260906.md">报告</a></p>'''
page=page.replace('10:01 CST',snapshot_time).replace('../PI05_THREE_RUNS_READONLY_REFRESH_20260906.md','../PI05_BC_SEEDS_TIMING_CONTEXT_STORAGE_DISCUSSION_20260906.md' if args.input!='PI05_THREE_RUNS_READONLY_REFRESH_20260906.json' else '../PI05_THREE_RUNS_READONLY_REFRESH_20260906.md')
for name,title in [('overview','完整总览'),('bc_comparison','BC对照'),('success','固定评估'),('training','训练采集'),('optimization','优化/DVAC'),('resources','资源')]:page+=f'<button onclick="show(\'{name}\')">{title}</button>'
for name,svg in svgs.items():page+=f'<section id="{name}" class="{("active" if name=="bc_comparison" else "")}">{svg}<p><a href="{name}.png">独立PNG</a></p></section>'
page+='<script>function show(id){document.querySelectorAll("section").forEach(s=>s.classList.toggle("active",s.id===id))}</script></html>'
(out/'index.html').write_text(page,encoding='utf-8')
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(out)
