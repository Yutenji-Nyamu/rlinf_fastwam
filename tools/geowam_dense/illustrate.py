"""Redraw a completed trajectory with three manually chosen, frame-aligned observations."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def illustrate(episode,raw,out,steps):
    info=json.loads((episode/'complete.json').read_text())
    with (episode/'curve.csv').open() as f:rows=list(csv.DictReader(f))
    assert steps[0]<steps[1]<steps[2] and steps[1]==info['peak']['global_step']
    out.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(9.2,4.4),layout='constrained')
    gs=fig.add_gridspec(2,3,height_ratios=[1.3,1])
    fig.suptitle(info['id']+' · '+info['task'].replace('_',' ').title(),fontsize=12)
    fields=['position','action_step','video_frame','time_s','entropy','depth_valid','image_name']
    with (out/'frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for i,(label,step) in enumerate(zip(['Before','Peak','After'],steps)):
            r=rows[step-1];assert r['depth_valid']=='1' and int(r['action_step'])==step
            w.writerow(dict(r,position=label))
            with Image.open(raw/'images/head'/r['image_name']) as src:im=src.convert('RGB')
            im.save(out/f'{label.lower()}-original.png')
            ax=fig.add_subplot(gs[0,i]);ax.imshow(im);ax.axis('off');ax.set_title(f'{label} · Step {step}',fontsize=11)
            d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,25),fill='#19252c')
            d.text((8,5),f"{label}  |  Step {step}  |  Frame {r['video_frame']}  |  {float(r['time_s']):.1f}s",fill='white')
            im.save(out/f'{label.lower()}.png')
    a=np.arange(1,len(rows)+1);h=np.asarray([float(r['entropy']) if r['entropy'] else np.nan for r in rows])
    ax=fig.add_subplot(gs[1,:]);ax.plot(a,h,color='#1c638c',lw=1.3)
    for s in steps:ax.plot(s,h[s-1],'o',color='#c85227',ms=4)
    ax.set(xlim=(1,len(a)),xlabel='Recorded action step',ylabel='Action entropy (nats)');ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):fig.savefig(out/('figure.'+ext),dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('raw',type=Path);p.add_argument('output',type=Path)
    p.add_argument('steps',nargs=3,type=int);a=p.parse_args();illustrate(a.episode,a.raw,a.output,a.steps)
