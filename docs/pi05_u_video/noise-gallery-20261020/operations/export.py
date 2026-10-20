"""Reuse accepted dense-U selection and same-episode controller/frame mapping."""
import json,tarfile,shutil
from pathlib import Path
import numpy as np
import cv2
R=Path('/data/chenyiteng/results/pi05-noise-video/20261019-sz1-v2')
O=Path('/data/chenyiteng/exports/noise4-gallery-20261020')
assert not O.exists()or not any(O.iterdir());O.mkdir(parents=True,exist_ok=True)
tasks={};errors=[];coverage={}
for d in sorted((R/'batches').iterdir()):
 if not(d/'done.json').exists()or not(d/'validation.json').exists():continue
 try:
  assert json.loads((d/'validation.json').read_text())['passed']
  es=json.loads((d/'episodes.json').read_text());qs=[]
  for p in sorted(d.glob('query_[0-9][0-9][0-9].npz')):
   with np.load(p)as z:qs.append({k:z[k]for k in ['u','noise4_u','active','before','after']})
  for e in es:
   coverage[e['task']]=coverage.get(e['task'],0)+1
   slot=e['slot'];fm=json.loads((d/e['frame_map']).read_text());frames=fm['frames']
   xs=[];ys=[];us=[];qq=[];hh=[];sides={}
   for q,z in enumerate(qs):
    if not z['active'][slot]:continue
    cp=d/f'q{q:03d}_s{slot:02d}_controller.npz'
    if not cp.exists():continue
    with np.load(cp)as c:
     dist={side:float(np.linalg.norm(np.diff(c[side+'_original_path'],axis=0),axis=1).sum())for side in ['left','right']}
    side=max(dist,key=dist.get);sides[q]=side
    mapped=[f[side+'_h_projection']for f in frames if f['query']==q and f.get(side+'_h_projection')is not None]
    if not mapped:continue
    upper=min(49,int(np.floor(max(mapped)+1e-5)))
    for h in range(upper+1):
     xs.append(int(z['before'][slot])+h);ys.append(float(z['noise4_u'][slot,h]));us.append(float(z['u'][slot,h]));qq.append(q);hh.append(h)
   if len(ys)<30:continue
   y=np.array(ys);pk=int(y.argmax());q=qq[pk];h=hh[pk];side=sides[q]
   near=[f for f in frames if f['query']==q and f.get(side+'_h_projection')is not None]
   f=min(near,key=lambda f:abs(f[side+'_h_projection']-h));delta=abs(f[side+'_h_projection']-h)
   edge=min(pk,len(y)-1-pk);contrast=float(y[pk]/max(np.median(y),1e-9));width=int(np.sum(y>y[pk]*.6))
   score=float(min(contrast,12)*(min(edge,30)/30)/(1+width/60)/(1+delta))
   rec=dict(**e,batch_name=d.name,remote_dir=str(d),action=xs,noise4=ys,u=us,query=qq,h=hh,query_sides=sides,
            peak_action=xs[pk],peak_index=pk,contrast=contrast,score=score,peak_h_error=delta,edge=edge)
   tasks.setdefault(e['task'],[]).append(rec)
 except Exception as ex:errors.append(dict(batch=d.name,error=repr(ex)))
selected=[]
for task,rows in sorted(tasks.items()):
 # Prefer completed successes, then interior peaks with accurate mapped frames.
 valid=[r for r in rows if r['success']]or rows
 candidates=sorted(valid,key=lambda r:(r['edge']>=5 and r['peak_h_error']<=2.5,r['score']),reverse=True)
 chosen=[candidates[0]]
 if len(candidates)>1 and candidates[1]['success']and candidates[1]['score']>=max(.5,candidates[0]['score']*.8)and candidates[1]['actual']!=candidates[0]['actual']:
  chosen.append(candidates[1])
 selected.extend(chosen)
for i,c in enumerate(selected,1):
 c['id']=f'N{i:02d}';d=Path(c['remote_dir']);dest=O/c['id'];dest.mkdir()
 y=np.array(c['noise4']);pk=c['peak_index'];cut=np.quantile(y,.35)+(y[pk]-np.quantile(y,.35))*.3
 a=pk
 while a>0 and y[a-1]>cut:a-=1
 b=pk
 while b<len(y)-1 and y[b+1]>cut:b+=1
 left=max(0,min(a-3,pk-9));right=min(len(y)-1,max(b+3,pk+9))
 positions=sorted(set([max(0,left-12),left,max(0,pk-5),pk,min(len(y)-1,pk+5),right,min(len(y)-1,right+14)]))
 fm=json.loads((d/c['frame_map']).read_text());options=[]
 for idx in positions:
  q=c['query'][idx];h=c['h'][idx];side=c['query_sides'][q]
  near=[f for f in fm['frames']if f['query']==q and f.get(side+'_h_projection')is not None]
  frame=min(near,key=lambda f:abs(f[side+'_h_projection']-h))
  options.append(dict(index=idx,action=c['action'][idx],query=q,side=side,frame=frame['frame'],tick=frame['tick'],
                      mapped_action=c['action'][idx]-h+frame[side+'_h_projection'],
                      h_error=abs(frame[side+'_h_projection']-h),peak=idx==pk,image=f"frame_{frame['frame']:05d}.jpg"))
 cap=cv2.VideoCapture(str(d/c['video']))
 wanted={v['frame']for v in options};n=0
 while wanted:
  ok,im=cap.read()
  if not ok:break
  if n in wanted:cv2.imwrite(str(dest/f'frame_{n:05d}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,94]);wanted.remove(n)
  n+=1
 cap.release();assert not wanted,(c['id'],wanted)
 shutil.copy2(d/c['video'],dest/'episode.mp4')
 shutil.copy2(d/c['frame_map'],dest/'frame_map.json')
 c.update(options=options,selected_indices=[positions.index(left),positions.index(pk),positions.index(right)],
          fps=fm['fps'],mapping_note=fm.get('h_mapping'),selection='Illustrative peak/transition selection, success preferred; not an unbiased metric.')
 (dest/'series.json').write_text(json.dumps(c))
(O/'manifest.json').write_text(json.dumps(dict(coverage=coverage,tasks=len(tasks),selected=selected,errors=errors,
 missing_tasks=sorted({p.stem.rsplit('_b',1)[0]for p in(R/'configs').glob('*_b[01].json')}-set(tasks))),ensure_ascii=False))
with tarfile.open(O.with_suffix('.tar.gz'),'w:gz')as t:t.add(O,arcname='.')
print(json.dumps(dict(tasks=len(tasks),selected=len(selected),coverage=coverage,errors=errors,archive=str(O.with_suffix('.tar.gz')),bytes=O.with_suffix('.tar.gz').stat().st_size)))
