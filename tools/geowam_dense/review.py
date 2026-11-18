"""Compare episodes for figure selection using their dense curves and recorded frames."""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw


def review(root):
    out=root/'review';out.mkdir(exist_ok=True)
    rows=[]
    for f in sorted(root.glob('G*/complete.json')):
        j=json.loads(f.read_text())
        if not j['peak']:continue
        with np.load(f.parent/'samples.npz') as z:
            v=z['first_step_entropy'];t=z['time_s'];state=z['state']
            dt=float(np.nanpercentile(v,90)-np.nanpercentile(v,10))
            p=j['peak'];s=p['plotted_step']-1
            events=np.flatnonzero(np.diff(state[:,-1])!=0)+1
            near=float(np.min(abs(t[events]-t[s]))) if len(events) else None
            margin=p['peak_gap']/dt if dt>0 and p['peak_gap'] is not None else None
        rows.append(dict(id=j['id'],task=j['task'],recording=j['episode'],split=j['split'],
                         observations=j['observations'],valid=j['valid'],global_peak=p['global_step'],
                         before=p['frame_steps'][0],peak=p['plotted_step'],after=p['frame_steps'][2],
                         other_peak=p['other_peak_step'],gap_nats=p['peak_gap'],gap_over_interdecile_range=margin,
                         nearest_gripper_transition_s=near))
    if not rows:return
    with (out/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    focus={}
    for task in ['plug_charger','sort_block']:
        candidates=[r for r in rows if r['task']==task]
        ordered=sorted(candidates,key=lambda r:(r['global_peak']==r['peak'],
            min(r['peak']-r['before'],r['after']-r['peak'])>=5,
            r['gap_over_interdecile_range'] if r['gap_over_interdecile_range'] is not None else -1),reverse=True)
        primary='G08' if task=='plug_charger' else 'G02'
        selection=([r for r in ordered if r['id']==primary]+[r for r in ordered if r['id']!=primary][:3])
        focus[task]=[r['id'] for r in selection]
        if not selection:continue
        sheet=Image.new('RGB',(1024,444*((len(selection)+1)//2)),'white')
        draw=ImageDraw.Draw(sheet)
        for pos,r in enumerate(selection):
            x=512*(pos%2);y=444*(pos//2)
            with Image.open(root/r['id']/'figure.png') as im:
                thumb=ImageOps.contain(im.convert('RGB'),(508,294));sheet.paste(thumb,(x,y+18))
            draw.text((x+3,y+3),f"{r['id']}  {r['recording']}  |  {r['valid']}/{r['observations']} valid",fill='black')
            for side,name in enumerate(['before','peak','after']):
                with Image.open(root/r['id']/(name+'-original.png')) as im:
                    sheet.paste(ImageOps.contain(im.convert('RGB'),(169,130)),(x+side*170,y+300))
        sheet.save(out/(task+'.jpg'),quality=90)
    (out/'focus.json').write_text(json.dumps(focus))
    print(len(rows),focus)
if __name__=='__main__':review(Path(sys.argv[1]))
