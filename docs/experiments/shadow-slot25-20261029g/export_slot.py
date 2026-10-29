import json,os,subprocess,tarfile,csv,shutil
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import cv2,numpy as np
B=Path('/home/chenyiteng/results/pi05-distill/fqlfix-shadow-sz1-20261029/teacher-shadow/batch0')
O=Path('/home/chenyiteng/deployment-20261029/shadow-slot25-1029g');O.mkdir(exist_ok=True)
fm=json.loads((B/'episode_25_frames.json').read_text())
episode=next(e for e in json.loads((B/'episodes.json').read_text()) if e['slot']==25)
assert episode['actual']==100112571 and episode['success'] and episode['submitted']==100
rows=[r for r in csv.DictReader((B/'signals.csv').open()) if int(r['slot'])==25]
assert len(rows)==100
with (O/'signals.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
shutil.copyfile(B/'episode_25_frames.json',O/'frame_map.json')
shutil.copyfile(B/'episode_25_control.jsonl',O/'control.jsonl')
shutil.copyfile(B/'episode_25.mp4',O/'episode-original.mp4')
paths={}
for q in range(5):
    p=B/f'q{q:03d}_s25_controller.npz';shutil.copyfile(p,O/p.name)
    with np.load(p) as z:
        paths[q]={side:float(np.linalg.norm(np.diff(z[side+'_original_path'],axis=0),axis=1).sum()) for side in ['left','right']}
mapped=[]
for f in fm['frames']:
    q=f['query'];r=dict(f)
    if q>=0:
        side=max(paths[q],key=paths[q].get);h=f.get(side+'_h_projection')
        r.update(active_arm=side,action_projection=None if h is None else q*20+h,
                 left_action_projection=None if f.get('left_h_projection') is None else q*20+f['left_h_projection'],
                 right_action_projection=None if f.get('right_h_projection') is None else q*20+f['right_h_projection'])
    mapped.append(r)
cap=cv2.VideoCapture(str(B/'episode_25.mp4'))
assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(mapped)
# Every 20th video frame for visual stage inspection, plus the actual final frame.
inspect_indices=sorted(set(range(0,len(mapped),20))|{len(mapped)-1})
for i in inspect_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();assert ok
    cv2.imwrite(str(O/f'preview-{i:03d}.jpg'),img,[cv2.IMWRITE_JPEG_QUALITY,94])
cap.release()
ff=subprocess.run(['ffmpeg','-v','error','-y','-i',str(B/'episode_25.mp4'),'-c:v','libx264','-crf','18','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(O/'episode-h264.mp4')],capture_output=True,text=True)
assert ff.returncode==0,ff.stderr[-500:]
meta={'episode':episode,'frames':mapped,'active_arm_path_lengths':paths,'fps':fm['fps'],'duration_seconds':len(mapped)/fm['fps'],
      'signal_length':100,'source_video':str(B/'episode_25.mp4'),'mapping':'same accepted per-arm monotone controller path projection; approximate, not frame number equals model action; last block terminates on success'}
(O/'manifest.json').write_text(json.dumps(meta,indent=2))
with tarfile.open(O/'slot25-local.tar','w') as t:
    for p in O.iterdir():
        if p.name!='slot25-local.tar':t.add(p,arcname=p.name)
print(json.dumps({'output':str(O),'files':len(list(O.iterdir())),'bytes':(O/'slot25-local.tar').stat().st_size,'arm_paths':paths,'frames':len(mapped)}))
