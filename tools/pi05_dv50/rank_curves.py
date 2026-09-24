"""Fixed shape ranking plus a self-contained query/video gallery. No causal labels."""
import html
import json
from pathlib import Path
import sys
import subprocess
import shutil
import numpy as np


def rank(values):
    y=np.asarray(values,dtype=float)
    if len(y)<4 or not np.isfinite(y).all() or y.max()<=0:
        return {'score':0.,'reason':'short, zero or invalid','smooth':y.tolist()}
    smooth=np.array([np.median(y[max(0,i-1):i+2]) for i in range(len(y))])
    lo,hi=np.quantile(smooth,[.2,.8]);amplitude=hi-lo
    if amplitude<=max(float(y.mean())*.05,1e-12):
        return {'score':0.,'reason':'nearly flat','smooth':smooth.tolist()}
    high=smooth>lo+.6*amplitude
    padded=np.r_[False,high,False].astype(int);starts=np.where(np.diff(padded)==1)[0];ends=np.where(np.diff(padded)==-1)[0]
    sustained=sum((ends-starts)>=2);transitions=int(np.abs(np.diff(high.astype(int))).sum())
    roughness=float(np.median(np.abs(y-smooth)))/(amplitude+1e-12)
    reason='candidate' if 1<=sustained<=3 and transitions<=6 and roughness<.5 else 'noisy or no sustained region'
    score=float(amplitude/(abs(lo)+.1*amplitude+1e-12)/(1+roughness*4)/(1+abs(sustained-1))) if reason=='candidate' else 0.
    mass=y/y.sum();top=float(np.sort(mass)[-max(1,int(np.ceil(.2*len(y)))):].sum())
    return {'score':score,'reason':reason,'high_regions':int(sustained),'transitions':transitions,
        'roughness':roughness,'top20_mass':top,'ess_fraction':float(1/(np.square(mass).sum()*len(y))),
        'entropy_normalized':float(-(mass*np.log(mass+1e-30)).sum()/np.log(len(y))), 'smooth':smooth.tolist()}


def main(root):
    root=Path(root);rows=[]
    for done in sorted(root.glob('batches/*/done.json')):
        d=done.parent;eps=json.loads((d/'episodes.json').read_text());queries=json.loads((d/'queries.json').read_text())
        # OpenCV's portable mp4v writer is not supported by every browser. Convert only
        # finalized previews; atomic replacement preserves any concurrent reader.
        if not (d/'browser-video.json').exists():
            ffmpeg=shutil.which('ffmpeg')
            if not ffmpeg:
                import imageio_ffmpeg
                ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
            for ep in eps:
                video=d/ep['video'];tmp=video.with_name(video.stem+'.h264.mp4')
                subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(video),
                    '-an','-c:v','libx264','-threads','1','-preset','fast','-crf','23',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],check=True,timeout=60)
                tmp.replace(video)
            (d/'browser-video.json').write_text(json.dumps({'codec':'H264','episodes':len(eps)}))
        for ep in eps:
            qs=[q for q in queries if q['slot']==ep['slot']]
            # Ranking excludes the terminal-success chunk with unknown executed prefix.
            exact=[q['dv_mean'] for q in qs if q['executed_prefix_known']]
            rows.append({**ep,**rank(exact),'batch_dir':str(d.relative_to(root)),
                 'queries':qs,'raw':[q['dv_mean'] for q in qs],
                 'rank_scope':'known fully executed chunks; terminal-success prediction shown separately'})
    rows.sort(key=lambda r:r['score'],reverse=True)
    (root/'ranking.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    document='''<!doctype html><meta charset="utf-8"><title>π0.5 DV50</title>
<style>body{font:15px system-ui;background:#f5f6f9;margin:25px;color:#202838}select,label{margin:8px}article{background:white;padding:18px;margin:14px 0;border-radius:12px}canvas{width:95%;max-width:900px;height:230px}video{width:480px;max-width:100%}.note{color:#667}h2{font-size:18px}</style>
<h1>π0.5 · 50任务DV轨迹</h1><p>原始DV与固定3-chunk中值线。视频每个决策前后采一帧，4 fps播放，非实时动作录像。高DV只表示模型预测分歧。</p>
<select id="task"><option value="">全部任务</option></select><label><input type="checkbox" id="only" checked>仅候选</label><label><input type="checkbox" id="raw" checked>原线</label><label><input type="checkbox" id="smooth" checked>中值线</label><div id="count"></div><main></main>
<script>const rows=PAYLOAD; const tasks=[...new Set(rows.map(x=>x.task))].sort();tasks.forEach(t=>task.add(new Option(t,t)));
function draw(){const chosen=rows.filter(r=>(!task.value||r.task===task.value)&&(!only.checked||r.score>0));count.textContent=`已完成轨迹 ${rows.length}；符合筛选 ${chosen.length}；显示前100条（全量见ranking.json）`;document.querySelector('main').replaceChildren();for(const r of chosen.slice(0,100)){let a=document.createElement('article');a.innerHTML=`<h2>${r.task} · seed ${r.actual} · ${r.success?'成功':'失败'}</h2><p class="note">${r.reason} · score ${r.score.toFixed(2)} · 原动作chunk；成功终止chunk的执行前缀未知，以橙色点表示。</p><canvas width="1000" height="240"></canvas><br><video controls preload="none" src="${r.batch_dir}/${r.video}"></video><p class="note">点曲线可跳至该chunk后的预览帧</p>`;document.querySelector('main').append(a);let c=a.querySelector('canvas'),ctx=c.getContext('2d'),y=r.raw,max=Math.max(...y,1e-12),n=y.length;ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(40,10);ctx.lineTo(40,205);ctx.lineTo(980,205);ctx.stroke();ctx.fillText(max.toExponential(2),2,15);ctx.fillText('0',20,208);function line(v,col,width){ctx.strokeStyle=col;ctx.lineWidth=width;ctx.beginPath();v.forEach((z,i)=>{const x=40+i*940/Math.max(n-1,1),p=205-z/max*190;i?ctx.lineTo(x,p):ctx.moveTo(x,p)});ctx.stroke()}if(raw.checked)line(y,'#aac4ef',2);if(smooth.checked)line(r.smooth,'#205dab',3);r.queries.forEach((q,i)=>{if(!q.executed_prefix_known){ctx.fillStyle='#df7d22';ctx.beginPath();ctx.arc(40+i*940/Math.max(n-1,1),205-y[i]/max*190,5,0,7);ctx.fill()}});c.onclick=e=>{const i=Math.max(0,Math.min(n-1,Math.round(((e.offsetX/c.clientWidth*1000)-40)/940*(n-1))));a.querySelector('video').currentTime=r.queries[i].post_frame/4};}}
[task,only,raw,smooth].forEach(x=>x.onchange=draw);draw();</script>'''.replace('PAYLOAD',payload)
    (root/'index.html').write_text(document,encoding='utf-8')
    print(json.dumps({'episodes':len(rows),'candidates':sum(r['score']>0 for r in rows)}))


if __name__=='__main__':main(sys.argv[1])
