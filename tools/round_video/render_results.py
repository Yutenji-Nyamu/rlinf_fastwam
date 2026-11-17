"""Render saved round timings and continuous Dojo signal/video examples."""
import csv,json,zipfile
from pathlib import Path
import numpy as np


def timing(root,dest,memory_log):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=json.loads((root/'rounds.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    gpu=[json.loads(s)for s in memory_log.read_text().splitlines()]if memory_log.exists()else[]
    rows=[]
    for k in [1,2,4,8]:
        rr=[r for r in records if r['K']==k and not r['warmup']]
        if len(rr)!=3:raise ValueError(f'K{k} has {len(rr)}/3 measured rounds')
        row={'K':k,'wall_s':float(np.mean([r['wall_seconds']for r in rr]))}
        row['wall_std_s']=float(np.std([r['wall_seconds']for r in rr],ddof=1))
        for key in ['environment','policy','candidates','entropy_weights','update']:
            row[key+'_s']=float(np.mean([r['seconds'][key]for r in rr]))
        row['other_s']=row['wall_s']-sum(row[x+'_s']for x in ['environment','policy','candidates','entropy_weights','update'])
        readings=[float(x['reading'].split(',')[0])/1024 for x in gpu if x['reading'] and any(r['started_unix']<=x['time']<=r['ended_unix']for r in rr)]
        row['peak_card_gib']=max(readings)if readings else None
        row['extra_sampling_fraction']=row['candidates_s']/row['wall_s'];rows.append(row)
    for r in rows:r.update(extra_round_s=r['wall_s']-rows[0]['wall_s'],relative_round_increase=r['wall_s']/rows[0]['wall_s']-1)
    with(dest/'round-cost.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained');bottom=np.zeros(4)
    for key,label in [('environment','Environment/reset'),('policy','Main inference'),('candidates','Extra samples'),('entropy_weights','Entropy/weights'),('update','Update'),('other','Other')]:
        v=np.array([r[key+'_s']/60 for r in rows]);ax.bar(range(4),v,bottom=bottom,label=label);bottom+=v
    ax.set(xticks=range(4),xticklabels=['K1','K2','K4','K8'],ylabel='Minutes per complete training round');ax.legend(fontsize=8,ncol=3)
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'round-cost.{ext}',dpi=180)
    k4=rows[2]
    (dest/'RESULT.md').write_text(f"# Complete-round K cost\n\n{protocol['algorithm']}; {protocol['task']}; {protocol['parallel']} environments; {protocol['denoise_steps']} denoising steps; {protocol['optimizer_steps']} updates of {protocol['global_batch']} samples. Three measured rounds per K after one warmup.\n\nK4 versus K1: {k4['extra_round_s']:.2f} s ({100*k4['relative_round_increase']:.2f}%). Extra candidate sampling occupies {100*k4['extra_sampling_fraction']:.2f}% of the K4 round.\n\nSynchronous measured wall time; module totals and full-device memory readings retained. Initialization and optional evaluation/checkpointing are outside training rounds.\n")
    for n in ['rounds.json','protocol.json','complete.json']:(dest/n).write_bytes((root/n).read_bytes())
    if memory_log.exists():(dest/'gpu-memory.jsonl').write_bytes(memory_log.read_bytes())


def video(root,dest):
    import cv2
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    task=root/'dojo/put_bottles_into_dustbin';episodes=json.loads((task/'episodes.json').read_text())
    for slot in [0,11,9]:
        name=f'D-B{slot:02d}';target=dest/name;target.mkdir(exist_ok=True)
        source_video=root/'videos'/f'{name}.mp4';(target/source_video.name).write_bytes(source_video.read_bytes())
        mapfile=root/'videos'/f'{name}-frames.csv';(target/mapfile.name).write_bytes(mapfile.read_bytes())
        frames=list(csv.DictReader(mapfile.open()));xs=[];ys=[];qs=[]
        for p in sorted((task/'queries').glob('q*.npz')):
            with np.load(p,allow_pickle=False)as z:
                lo=int(z['before'][slot]);hi=int(z['after'][slot]);count=hi-lo
                if count<=0:continue
                xs.extend(range(lo+1,hi+1));ys.extend(z['entropy'][slot,:count]);qs.extend([int(p.stem[1:])]*count)
                np.savez_compressed(target/p.name,model_actions=z['noise4_model_actions'][slot],env_actions=z['noise4_env_actions'][slot],initial_noise=z['noise4_initial_noise'][slot],entropy=z['entropy'][slot],before=lo,after=hi)
            with np.load(task/'observations'/p.name,allow_pickle=False)as z:
                np.savez_compressed(target/(p.stem+'-obs.npz'),**{k:z[k][slot]for k in z.files})
        peak=int(np.argmax(ys));chosen=[max(0,peak-20),peak,min(len(xs)-1,peak+20)]
        cap=cv2.VideoCapture(str(source_video));fig=plt.figure(figsize=(9,5),layout='constrained');gs=fig.add_gridspec(2,3);marks=[]
        for j,idx in enumerate(chosen):
            f=min(frames,key=lambda f:abs(int(f['action_step'])-xs[idx]));cap.set(cv2.CAP_PROP_POS_FRAMES,int(f['frame']));ok,img=cap.read()
            if not ok:raise RuntimeError('Missing selected execution frame')
            label='ABC'[j];cv2.imwrite(str(target/f'{label}.png'),img);ax=fig.add_subplot(gs[0,j]);ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));ax.axis('off');ax.set_title(f'{label}: action {xs[idx]}')
            marks.append(dict(label=label,action=xs[idx],query=qs[idx],frame=int(f['frame'])))
        cap.release();ax=fig.add_subplot(gs[1,:]);ax.plot(xs,ys,color='#227eab',lw=1.3);ax.set(xlabel='Executed action step',ylabel='Action entropy',title=name)
        for m in marks:ax.axvline(m['action'],ls=':',color='#777')
        for ext in ['png','pdf','svg']:fig.savefig(target/f'{name}.{ext}',dpi=180)
        plt.close(fig);np.savez_compressed(target/'curve.npz',action=xs,entropy=ys,query=qs)
        (target/'selected-frames.json').write_text(json.dumps(marks,indent=2));(target/'episode.json').write_text(json.dumps(episodes[slot],indent=2))
    (dest/'initial-match.json').write_bytes((task/'initial-match.json').read_bytes())
    aligned=[json.loads(p.read_text())for p in sorted((task/'alignment').glob('*.json'))]
    (dest/'query-alignment.json').write_text(json.dumps(aligned,indent=2))
    (dest/'RESULT.md').write_text('# Dojo continuous execution frames\n\nD-B00, D-B11 and D-B09: original layout IDs and policy RNG; K4 recomputed on every reached observation. New curves are paired to this recorded execution. MP4 is one frame per native action, played at 25 fps; frame/action/query CSV is the alignment reference. A/B/C mark peak minus 20 actions, peak, and peak plus 20 actions.\n')
