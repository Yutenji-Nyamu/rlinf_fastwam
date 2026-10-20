"""CPU-only export: raw executed-action U, native frame mapping, selected media."""
import json,time,tarfile,subprocess,collections
from pathlib import Path
import numpy as np
import cv2
R=Path('/data/chenyiteng/results/dojo-u-video/20261018-v1/tasks')
O=Path('/data/chenyiteng/exports/dojo-u-gallery-20261020')
O.mkdir(parents=True,exist_ok=True)
groups=collections.defaultdict(list);runs=[];errors=[]
for run in sorted(R.iterdir()):
 if not run.is_dir():continue
 epfiles=list(run.glob('episodes/*/episode.json'))
 result=json.loads((run/'result.json').read_text())if(run/'result.json').exists()else None
 rows=[]
 for p in epfiles:
  ep=json.loads(p.read_text());ep['directory']=str(p.parent);rows.append(ep)
  groups[ep['task']].append(ep)
 runs.append({'run':run.name,'episodes':len(rows),'successes':sum(e['success']for e in rows),'result':result})
selected=[];stats={}
for task,episodes in sorted(groups.items()):
 candidates=[];medians=[];peaks=[]
 for e in episodes:
  d=Path(e['directory']);n=e['executed_actions']
  try:
   ys=[]
   for q in range((n+49)//50):
    with np.load(d/f'query_{q:04d}.npz')as z:ys.extend(z['u'].tolist())
   y=np.asarray(ys[:n],dtype=float)
   if len(y)!=n or n<15 or not np.isfinite(y).all():continue
   # Interior peak with a visible local excursion; raw curve is never smoothed.
   local=[i for i in range(8,n-8)if y[i]>=y[i-1]and y[i]>=y[i+1]]
   if not local:local=[int(y.argmax())]
   def prominence(i):
    lo=max(0,i-35);hi=min(n,i+36)
    return float(y[i]-max(np.min(y[lo:i+1]),np.min(y[i:hi])))
   pk=max(local,key=prominence);prom=prominence(pk)
   med=float(np.median(y));span=float(np.quantile(y,.95)-np.quantile(y,.05))
   width=int(np.sum(y>med+.6*(y[pk]-med)))
   score=(prom/max(med,1e-9))/(1+width/60)
   candidates.append(dict(**e,u=y.tolist(),action=list(range(n)),peak_index=pk,
     peak_u=float(y[pk]),median_u=med,prominence=prom,score=score,span=span))
   medians.append(med);peaks.append(float(y.max()))
  except (OSError,KeyError,ValueError)as ex:errors.append({'episode':str(d),'error':repr(ex)})
 stats[task]={'episodes':len(episodes),'successes':sum(e['success']for e in episodes),
  'usable_signal':len(candidates),'episode_median_u':float(np.median(medians))if medians else None,
  'median_peak_u':float(np.median(peaks))if peaks else None}
 pool=[c for c in candidates if c['success']]or candidates
 pool.sort(key=lambda c:c['score'],reverse=True)
 if not pool:continue
 chosen=[pool[0]]
 for c in pool[1:]:
  if c['success']and c['source_layout']!=chosen[0]['source_layout']:
   chosen.append(c);break
 selected.extend(chosen)
for i,c in enumerate(selected,1):
 c['id']=f'D{i:02d}';dest=O/c['id'];dest.mkdir(exist_ok=True);d=Path(c['directory'])
 actions=[json.loads(x)for x in(d/'actions.jsonl').read_text().splitlines()]
 frames=[json.loads(x)for x in(d/'frames.jsonl').read_text().splitlines()]
 assert [a['action_index']for a in actions]==c['action']
 y=np.array(c['u']);pk=c['peak_index'];cut=c['median_u']+.3*(y[pk]-c['median_u'])
 left=pk
 while left>0 and y[left-1]>cut:left-=1
 right=pk
 while right<len(y)-1 and y[right+1]>cut:right+=1
 positions=[max(0,min(left-3,pk-10)),pk,min(len(y)-1,max(right+3,pk+10))]
 c['selected_frames']=[]
 for letter,idx in zip('ABC',positions):
  a=actions[idx];row={'label':letter,'action':idx,'query':a['query'],'h':a['h'],'u':c['u'][idx],'cameras':{}}
  for cam in c['videos']:
   matches=[f for f in frames if f['camera']==cam and f['query']==a['query']and f['h']==a['h']]
   assert matches,(c['id'],cam,idx)
   f=min(matches,key=lambda f:abs(f['control_tick']-a['end_tick']))
   name=f'{letter}_{cam}.jpg';cap=cv2.VideoCapture(c['videos'][cam]['path'])
   cap.set(cv2.CAP_PROP_POS_FRAMES,f['frame']);ok,im=cap.read();cap.release();assert ok
   cv2.imwrite(str(dest/name),im,[cv2.IMWRITE_JPEG_QUALITY,93])
   row['cameras'][cam]=dict(f,image=name)
  c['selected_frames'].append(row)
 c['fps']=c['videos']['cam_head']['fps']
 # Retain all native frame records as the alignment evidence.
 (dest/'frames.json').write_text(json.dumps(frames))
 (dest/'series.json').write_text(json.dumps(c))
 subprocess.run(['ffmpeg','-v','error','-y','-i',c['videos']['cam_head']['path'],
  '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','23','-pix_fmt','yuv420p',
  '-movflags','+faststart',str(dest/'episode.mp4')],check=True,timeout=120)
 print(json.dumps({'selected':c['id'],'task':c['task'],'success':c['success'],'actions':len(y)}),flush=True)
manifest={'time':time.time(),'tasks':stats,'runs':runs,'selected':selected,'errors':errors,
 'scope':'SZ3 formal task runs only; excludes SZ1 25-episode smoke',
 'selection':'Illustrative peaks; success preferred; zero-success tasks explicitly failure.',
 'mapping':'Native action index -> recorded query/h -> control tick -> same-episode camera frame.'}
(O/'manifest.json').write_text(json.dumps(manifest))
with tarfile.open(O.with_suffix('.tar.gz'),'w:gz')as t:t.add(O,arcname='.')
print(json.dumps({'archive':str(O.with_suffix('.tar.gz')),'bytes':O.with_suffix('.tar.gz').stat().st_size,'tasks':len(stats),'selected':len(selected),'episodes':sum(s['episodes']for s in stats.values()),'successes':sum(s['successes']for s in stats.values()),'errors':errors}))
