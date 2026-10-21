"""Reuse accepted dense-U selection and same-episode controller/frame mapping."""
import json,tarfile,shutil,time,subprocess,sys,collections
from pathlib import Path
import numpy as np
import cv2
R=Path('/data/chenyiteng/results/pi05-attn-objects/20261020-v1')
full='--full' in sys.argv
O=Path('/data/chenyiteng/exports/attn-objects-gallery-20261021' if full else '/data/chenyiteng/exports/attn-objects-preview-20261020')
deadline=time.monotonic()+1200
while not all((R/'batches'/b/'validation.json').exists() for b in ('turn_switch_b0','adjust_bottle_b1')):
 assert time.monotonic()<deadline, 'First batches not accepted before preview timeout'
 time.sleep(10)
assert not O.exists()or not any(O.iterdir());O.mkdir(parents=True,exist_ok=True)
tasks={};errors=[];coverage={};successes=collections.Counter();invalid_signals=[]
batches=sorted((R/'batches').iterdir()) if full else [R/'batches'/b for b in ('adjust_bottle_b1','turn_switch_b0')]
for d in batches:
 if not(d/'done.json').exists()or not(d/'validation.json').exists():continue
 try:
  assert json.loads((d/'validation.json').read_text())['passed']
  es=json.loads((d/'episodes.json').read_text());qs=[]
  for p in sorted(d.glob('query_[0-9][0-9][0-9].npz')):
   with np.load(p)as z:qs.append({k:z[k]for k in ['u','noise4_u','active','before','after']})
  for q,z in enumerate(qs):
   with np.load(d/f'query_{q:03d}_attention.npz') as a:
    for key in ('I_direct','I_via_visual','I_total'):z[key]=a[key][:,9,:,0]
    z['dv']=a['dv']
  for e in es:
   coverage[e['task']]=coverage.get(e['task'],0)+1
   successes[e['task']]+=int(e['success'])
   if full and not e['success']:continue
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
     xs.append(int(z['before'][slot])+h);ys.append(float(z['I_total'][slot,h]));us.append(float(z['u'][slot,h]));qq.append(q);hh.append(h)
   if len(ys)<30:continue
   y=np.array(ys)
   if not np.isfinite(y).all():
    invalid_signals.append(dict(task=e['task'],batch=d.name,slot=slot,reason='missing object-word score'))
    continue
   # Same raw curve; select an interior local excursion rather than a generic
   # first-block endpoint. This is illustrative selection, not a statistical test.
   peaks=[i for i in range(8,len(y)-8)if y[i]>=y[i-1]and y[i]>=y[i+1]]
   def prominence(i):
    return float(y[i]-max(min(y[max(0,i-30):i+1]),min(y[i:min(len(y),i+31)])))
   pk=max(peaks,key=prominence) if full and peaks else int(y.argmax())
   q=qq[pk];h=hh[pk];side=sides[q]
   near=[f for f in frames if f['query']==q and f.get(side+'_h_projection')is not None]
   f=min(near,key=lambda f:abs(f[side+'_h_projection']-h));delta=abs(f[side+'_h_projection']-h)
   edge=min(pk,len(y)-1-pk);contrast=float(y[pk]/max(np.median(y),1e-9));width=int(np.sum(y>y[pk]*.6))
   score=float(min(contrast,12)*(min(edge,30)/30)/(1+width/60)/(1+delta))
   rec=dict(**e,batch_name=d.name,remote_dir=str(d),action=xs,attention_total=ys,u=us,query=qq,h=hh,query_sides=sides,
            i_direct=[float(qs[q]['I_direct'][slot,h]) for q,h in zip(qq,hh)],
            i_via_visual=[float(qs[q]['I_via_visual'][slot,h]) for q,h in zip(qq,hh)],
            noise4=[float(qs[q]['noise4_u'][slot,h]) for q,h in zip(qq,hh)],
            dv=[float(qs[q]['dv'][slot,h]) for q,h in zip(qq,hh)],
            peak_action=xs[pk],peak_index=pk,contrast=contrast,score=score,peak_h_error=delta,edge=edge)
   if full:
    rec['score']=prominence(pk)/max(float(np.median(y)),1e-9)/(1+delta)
    rec['peak_selection']='largest interior local excursion of raw total attention'
   tasks.setdefault(e['task'],[]).append(rec)
  print(json.dumps(dict(processed=d.name,valid_episodes=sum(coverage.values()),successes=sum(successes.values()))),flush=True)
 except Exception as ex:errors.append(dict(batch=d.name,error=repr(ex)))
selected=[]
for task,rows in sorted(tasks.items()):
 # Prefer completed successes, then interior peaks with accurate mapped frames.
 valid=[r for r in rows if r['success']] if full else ([r for r in rows if r['success']]or rows)
 if not valid:continue
 candidates=sorted(valid,key=lambda r:(r['edge']>=5 and r['peak_h_error']<=2.5,r['score']),reverse=True)
 chosen=[candidates[0]]
 if len(candidates)>1 and candidates[1]['success']and candidates[1]['score']>=(candidates[0]['score']*.65 if full else max(.5,candidates[0]['score']*.8))and candidates[1]['actual']!=candidates[0]['actual']:
  chosen.append(candidates[1])
 selected.extend(chosen)
for i,c in enumerate(selected,1):
 c['id']=f'A{i:02d}';d=Path(c['remote_dir']);dest=O/c['id'];dest.mkdir()
 y=np.array(c['attention_total']);pk=c['peak_index'];cut=np.quantile(y,.35)+(y[pk]-np.quantile(y,.35))*.3
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
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i',str(d/c['video']),'-an','-c:v','libx264','-crf','20','-preset','veryfast','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'episode-h264.mp4')],check=True,timeout=120)
 shutil.copy2(d/c['frame_map'],dest/'frame_map.json')
 c.update(options=options,selected_indices=[positions.index(left),positions.index(pk),positions.index(right)],
          fps=fm['fps'],mapping_note=fm.get('h_mapping'),selection='Illustrative peak/transition selection; success-only for full gallery; not an unbiased metric.')
 meta=json.loads((d/f'query_{c["query"][pk]:03d}_attention.json').read_text())
 c['attention_labels']=meta.get('labels');c['object_annotation']=meta['annotations'][c['slot']]
 c['prompt_tokens']=meta['token_rows'][c['slot']]
 (dest/'series.json').write_text(json.dumps(c))
planned=sorted({json.loads(p.read_text())['task']for p in(R/'configs').glob('*.json')if not p.name.endswith('_seeds.json')})
(O/'manifest.json').write_text(json.dumps(dict(coverage=coverage,successes=dict(successes),tasks=len(tasks),selected=selected,errors=errors,
 preview=not full, planned_tasks=50, missing_tasks=sorted(set(planned)-set(coverage)),
 zero_success_tasks=[t for t in sorted(coverage)if not successes[t]],invalid_signals=invalid_signals),ensure_ascii=False))
with tarfile.open(O.with_suffix('.tar.gz'),'w:gz')as t:t.add(O,arcname='.')
print(json.dumps(dict(tasks=len(tasks),selected=len(selected),coverage=coverage,errors=errors,archive=str(O.with_suffix('.tar.gz')),bytes=O.with_suffix('.tar.gz').stat().st_size)))
