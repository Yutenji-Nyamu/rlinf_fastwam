"""Reuse the accepted per-arm projection and full-curve/phase-frame layout."""
import os,json,csv,base64,tarfile,shutil,xml.etree.ElementTree as ET
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np,cv2
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

O=Path('/home/chenyiteng/deployment-20261029/shadow-slot25-1029g')
m=json.loads((O/'manifest.json').read_text());rows=list(csv.DictReader((O/'signals.csv').open()))
x=np.array([int(r['action']) for r in rows]);u=np.array([float(r['noise4_u']) for r in rows]);d=np.array([float(r['teacher_student_rmse']) for r in rows])
font='/home/chenyiteng/deployment-20261029/fqlfix-shadow-sz1/msyh.ttc'
font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'svg.fonttype':'none','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
# Twelve chronological states: full coverage, major excursions and successful end.
targets=[None,8,16,25,38,46,54,59,70,78,87,'final']
captions=['初始','机械臂接近','夹爪靠近','对准瓶身','夹爪围住瓶身','保持夹持','夹持调整','准备搬移','向蓝垫移动','到达蓝垫上方','下放','成功结束']
selected=[];frames=m['frames']
cap=cv2.VideoCapture(str(O/'episode-original.mp4'))
for i,(target,caption) in enumerate(zip(targets,captions)):
    if target is None:f=frames[0];projected=0.;error=None
    elif target=='final':f=frames[-1];projected=f['action_projection'];error=None
    else:
        q=target//20;options=[f for f in frames if f['query']==q and f.get('action_projection') is not None]
        f=min(options,key=lambda r:abs(r['action_projection']-target));projected=f['action_projection'];error=abs(projected-target)
    cap.set(cv2.CAP_PROP_POS_FRAMES,f['frame']);ok,im=cap.read();assert ok
    name=f"frame-{i+1:02d}-{f['frame']:03d}.png";cv2.imwrite(str(O/name),im)
    selected.append(dict(letter=chr(65+i),caption=caption,target_action=target,action_projection=projected,
                         mapping_error=error,video_frame=f['frame'],video_time=f['frame']/m['fps'],
                         sim_seconds=f['sim_seconds'],image=name,active_arm=f.get('active_arm','left'),
                         left_action_projection=f.get('left_action_projection'),right_action_projection=f.get('right_action_projection')))
cap.release();assert len({s['video_frame'] for s in selected})==12
last=frames[-1]['action_projection'];physical=x<=last
m.update(selected_frames=selected,original_rho=float(spearmanr(u,d).statistic),reached_prefix_rho=float(spearmanr(u[physical],d[physical]).statistic),
         curve_axis='0-based submitted action positions; final controller-projected reach is 96.57; positions97-99 shown dashed',
         frame_selection='12 chronological stages with major local excursions; illustrative, not statistical sampling')
(O/'selected.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))

def curves(axes,zoom=False):
    for ax,v,col,label in zip(axes,[u,d],['#0072b2','#d55e00'],['四噪声相对分散度','师生动作距离 RMSE']):
        ax.plot(x[physical],v[physical],color=col,lw=1.9)
        ax.plot(x[~physical| (x==96)],v[~physical| (x==96)],color=col,lw=1.5,ls='--',alpha=.45)
        ax.set_ylabel(label);ax.set_ylim(0,max(v)*1.2);ax.set_xlim((60,99) if zoom else (0,99));ax.grid(axis='y',alpha=.16)
        ax.axvspan(last,99,color='#ddd',alpha=.5)
        for j,s in enumerate(selected):
            a=s['action_projection']
            if zoom and a<60:continue
            ax.axvline(a,color='#a9b4c2',lw=.6,ls=':')
            if ax is axes[0]:ax.text(a,max(v)*(1.07 if j%2==0 else 1.15),s['letter'],ha='center',fontsize=10)
    axes[-1].set_xlabel('提交动作位置（原图从0编号；A–L对应下方视频帧）')

fig=plt.figure(figsize=(16,13.5),layout='constrained');gs=fig.add_gridspec(5,4,height_ratios=[1.25,1.1,1.55,1.55,1.55])
axs=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:])];curves(axs)
for i,s in enumerate(selected):
    ax=fig.add_subplot(gs[2+i//4,i%4]);im=plt.imread(O/s['image']);ax.imshow(im);ax.axis('off')
    action='初始' if i==0 else f"动作≈{s['action_projection']:.1f}"
    ax.set_title(f"{s['letter']}  {s['caption']}\n{action} · 帧{s['video_frame']} · {s['video_time']:.2f}s",fontsize=10,pad=7)
fig.suptitle('slot 25 · pillbottle 成功轨迹\nClean CP140执行 / Uτ3 CP70旁路 · seed100112571 · 原100点ρ=0.430',fontsize=16)
fig.supxlabel('帧来自同一完整视频；动作位置按原控制器左臂路径近似映射。灰区为提前成功后未走到的末3个提交位置。',fontsize=10)
for ext in ['png','svg','pdf']:fig.savefig(O/('signal-and-12-frames.'+ext),dpi=150,bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(14,5.5),layout='constrained');curves(axs)
fig.suptitle('slot 25 · 两条完整原始信号（无平滑）',fontsize=15)
for ext in ['png','svg','pdf']:fig.savefig(O/('curves.'+ext),dpi=160,bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(13,5.2),layout='constrained');curves(axs,True)
fig.suptitle('搬向蓝垫的片段：距离在76附近升高，分散度在78附近出现峰',fontsize=14)
for ext in ['png','svg','pdf']:fig.savefig(O/('late-zoom.'+ext),dpi=150,bbox_inches='tight')
plt.close(fig)

# Editable Draw.io: individual images, labels and signal polylines.
mx=ET.Element('mxfile',host='app.diagrams.net',type='device');dia=ET.SubElement(mx,'diagram',name='slot25 全程＋12帧',id='slot25')
gm=ET.SubElement(dia,'mxGraphModel',dx='1800',dy='1500',grid='1',page='1',pageWidth='1800',pageHeight='1580');root=ET.SubElement(gm,'root')
ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0');seq=1
def box(text,xx,yy,ww,hh,style):
    global seq
    seq+=1;c=ET.SubElement(root,'mxCell',id=str(seq),value=text,style=style,vertex='1',parent='1')
    ET.SubElement(c,'mxGeometry',x=str(xx),y=str(yy),width=str(ww),height=str(hh),attrib={'as':'geometry'})
def line(points,color,width=2,dashed=False):
    global seq
    seq+=1;c=ET.SubElement(root,'mxCell',id=str(seq),value='',style=f'edgeStyle=none;endArrow=none;strokeColor={color};strokeWidth={width};rounded=0;'+('dashed=1;opacity=55;' if dashed else ''),edge='1',parent='1')
    g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    ET.SubElement(g,'mxPoint',x=str(points[0][0]),y=str(points[0][1]),attrib={'as':'sourcePoint'})
    ET.SubElement(g,'mxPoint',x=str(points[-1][0]),y=str(points[-1][1]),attrib={'as':'targetPoint'})
    a=ET.SubElement(g,'Array',attrib={'as':'points'})
    for xx,yy in points[1:-1]:ET.SubElement(a,'mxPoint',x=str(xx),y=str(yy))
textstyle='text;html=0;align=center;verticalAlign=middle;whiteSpace=wrap;fontFamily=Microsoft YaHei;fontSize=18;'
box('slot 25 · 成功 · seed100112571 · 100个提交动作位置 · 原指标ρ=0.430',50,12,1700,45,textstyle+'fontSize=27;fontStyle=1;')
for k,(v,col,name) in enumerate([(u,'#0072b2','四噪声相对分散度'),(d,'#d55e00','师生动作距离 RMSE')]):
    x0,y0,ww,hh=170,100+k*235,1540,175;vm=max(v)*1.2
    line([(x0,y0),(x0,y0+hh),(x0+ww,y0+hh)],'#444444',1)
    points=[(x0+float(a)/99*ww,y0+hh-float(b)/vm*hh)for a,b in zip(x,v)]
    line(points[:97],col);line(points[96:],col,dashed=True)
    box(name,3,y0+45,160,60,textstyle+'fontSize=17;')
    for a in range(0,100,20):box(str(a),x0+a/99*ww-20,y0+hh+4,40,22,textstyle+'fontSize=14;')
    for t in [0,.5,1]:box(f'{vm*t:.3f}',x0-62,y0+hh-t*hh-10,57,22,textstyle+'fontSize=13;align=right;')
    for s in selected:
        xx=x0+s['action_projection']/99*ww
        line([(xx,y0),(xx,y0+hh)],'#c1cbd5',.6)
        if k==0:box(s['letter'],xx-15,y0-23,30,22,textstyle+'fontSize=15;')
for i,s in enumerate(selected):
    xx=45+i%4*435;yy=610+i//4*300
    b64=base64.b64encode((O/s['image']).read_bytes()).decode()
    box('',xx,yy,400,235,'shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=1;aspect=fixed;image=data:image/png,'+b64+';')
    a='初始' if i==0 else f"动作≈{s['action_projection']:.1f}"
    box(f"{s['letter']} {s['caption']}  |  {a}\n帧{s['video_frame']} · {s['video_time']:.2f}s",xx,yy+238,400,55,textstyle+'fontSize=17;')
box('完整视频与曲线保持原始数据；动作—视频为路径近似映射，末帧≈96.57；最后3个未走到的提交位置用虚线表示。',40,1510,1720,45,textstyle+'fontSize=16;')
ET.ElementTree(mx).write(O/'slot25.drawio',encoding='utf-8',xml_declaration=True)
with tarfile.open(O/'slot25-figures.tar','w') as t:
    for p in O.iterdir():
        if p.name.startswith(('frame-','signal-and-','curves.','late-zoom.')) or p.name in ['selected.json','slot25.drawio']:
            t.add(p,arcname=p.name)
print(json.dumps({'selected':[(s['letter'],s['target_action'],s['video_frame'],round(s['action_projection'],2))for s in selected],'reached_prefix_points':int(physical.sum()),'reached_rho':m['reached_prefix_rho'],'archive_bytes':(O/'slot25-figures.tar').stat().st_size}))
