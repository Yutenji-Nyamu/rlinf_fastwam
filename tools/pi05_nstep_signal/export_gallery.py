"""Extend the accepted 10/20 U/K4 gallery; retain its controller-frame mapping.

CPU-only export. Source trajectories stay unchanged. Selection is illustrative,
success-first, based on N1/N10 peak clarity rather than signal correlation.
"""
import concurrent.futures,csv,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
import cv2
import numpy as np

R=Path(sys.argv[1]);O=Path(sys.argv[2]);O.mkdir(parents=True,exist_ok=True)
LIGHT=O/'all_tasks';LIGHT.mkdir(exist_ok=True)
KEYS=[f'{k}_{a}_{b}' for k in ['u','rmse'] for a,b in [(1,10),(1,5),(5,10)]]
KEYS += [f'noise{count}_{kind}_n{n}' for count in [4,3] for kind in ['relative','absolute'] for n in [1,5,10]]
KEYS += ['dv_n5','dv_n10']
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False))
tasks={};errors=[];coverage={};allstats=[]
for d in sorted((R/'batches').iterdir()):
 if not (d/'validation.json').exists() or not read(d/'validation.json').get('passed'):continue
 es=read(d/'episodes.json');task=es[0]['task'];coverage[task]={'episodes':len(es),'successes':sum(e['success'] for e in es)}
 ld=LIGHT/d.name;ld.mkdir(exist_ok=True)
 for name in ['episodes.json','queries.json','done.json','validation.json','signals.csv','nstep-parity.json','noise-capacity.json']:
  if (d/name).is_file():shutil.copy2(d/name,ld/name)
 for p in (R/'configs').glob(d.name+'*.json'):shutil.copy2(p,ld/p.name)
 qs=[]
 for p in sorted(d.glob('query_[0-9][0-9][0-9].npz')):
  with np.load(p) as z:qs.append({k:z[k] for k in KEYS+['active','before','after']})
 for e in es:
  slot=e['slot'];fm=read(d/e['frame_map']);frames=fm['frames'];sides={};xs=[];qq=[];hh=[];vals={k:[]for k in KEYS}
  for q,z in enumerate(qs):
   if not z['active'][slot]:continue
   cp=d/f'q{q:03d}_s{slot:02d}_controller.npz'
   if not cp.exists():continue
   with np.load(cp)as c:
    dist={s:float(np.linalg.norm(np.diff(c[s+'_original_path'],axis=0),axis=1).sum())for s in ['left','right']}
   side=max(dist,key=dist.get);sides[q]=side
   mapped=[f[side+'_h_projection']for f in frames if f['query']==q and f.get(side+'_h_projection')is not None]
   if not mapped:continue
   upper=min(49,int(z['after'][slot]-z['before'][slot])-1,int(np.floor(max(mapped)+1e-5)))
   for h in range(upper+1):
    xs.append(int(z['before'][slot])+h);qq.append(q);hh.append(h)
    for k in KEYS:vals[k].append(float(z[k][slot,h]))
  if len(xs)<5:
   errors.append({'task':task,'slot':slot,'reason':'too_few_controller_projected_points'});continue
  # Full-population projected-prefix scores for later statistical plotting.
  with (ld/f'slot-{slot:02d}-executed.csv').open('w',newline='')as f:
   wr=csv.writer(f);wr.writerow(['action','query','h']+KEYS)
   wr.writerows([x,q,h]+[vals[k][i]for k in KEYS]for i,(x,q,h)in enumerate(zip(xs,qq,hh)))
  shutil.copy2(d/e['frame_map'],ld/e['frame_map'])
  y=np.asarray(vals['u_1_10']);pk=int(y.argmax());q=qq[pk];h=hh[pk];side=sides[q]
  near=[f for f in frames if f['query']==q and f.get(side+'_h_projection')is not None]
  frame=min(near,key=lambda f:abs(f[side+'_h_projection']-h));delta=abs(frame[side+'_h_projection']-h)
  edge=min(pk,len(y)-1-pk);contrast=float(y[pk]/max(np.median(y),1e-9));width=int(np.sum(y>y[pk]*.6))
  score=float(min(contrast,12)*(min(edge,30)/30)/(1+width/60)/(1+delta))
  record=dict(**e,batch_name=d.name,remote_dir=str(d),action=xs,signals=vals,query=qq,h=hh,query_sides=sides,peak_index=pk,peak_action=xs[pk],peak_h_error=delta,score=score,edge=edge,contrast=contrast)
  tasks.setdefault(task,[]).append(record)
  allstats.append({'task':task,'slot':slot,'seed':e['actual'],'success':e['success'],'actions':len(xs),'selection_score':score,'peak_action':xs[pk]})
 print(json.dumps({'scanned':task,'episodes':len(es)}),flush=True)
selected=[]
for task,rows in sorted(tasks.items()):
 candidates=[r for r in rows if r['success']]or rows
 candidates=sorted(candidates,key=lambda r:(r['edge']>=5 and r['peak_h_error']<=2.5,r['score']),reverse=True)
 selected.extend(candidates[:2] if candidates[0]['success'] else candidates[:1])
for i,c in enumerate(selected,1):
 c['id']=f'S{i:02d}';d=Path(c['remote_dir']);dest=O/c['id'];dest.mkdir(exist_ok=True)
 fm=read(d/c['frame_map']);n=len(c['action']);pk=c['peak_index']
 positions=sorted(set([0,max(0,pk-12),pk,min(n-1,pk+12),n//2,n-1]))
 for frac in [.25,.75,.4,.6]:
  if len(positions)>=6:break
  positions=sorted(set(positions+[round((n-1)*frac)]))
 options=[]
 for idx in positions:
  q=c['query'][idx];h=c['h'][idx];side=c['query_sides'][q]
  near=[f for f in fm['frames']if f['query']==q and f.get(side+'_h_projection')is not None]
  fr=min(near,key=lambda f:abs(f[side+'_h_projection']-h))
  options.append({'index':idx,'action':c['action'][idx],'query':q,'side':side,'frame':fr['frame'],'tick':fr['tick'],'h_error':abs(fr[side+'_h_projection']-h),'image':f"frame_{fr['frame']:05d}.jpg",'peak':idx==pk})
 cap=cv2.VideoCapture(str(d/c['video']));wanted={p['frame']for p in options};j=0
 while wanted:
  ok,im=cap.read()
  if not ok:break
  if j in wanted:cv2.imwrite(str(dest/f'frame_{j:05d}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,94]);wanted.remove(j)
  j+=1
 cap.release();assert not wanted,(c['id'],wanted)
 shutil.copy2(d/c['frame_map'],dest/'frame_map.json')
 c.update(options=options,fps=fm['fps'],mapping_note=fm.get('h_mapping'),n1_dv=None,selection='success-first N1/N10 peak clarity, not correlation; illustrative only')
 write(dest/'series.json',c)
 rawdir=dest/'raw';rawdir.mkdir(exist_ok=True)
 # Save selected-slot raw actions, noises and internal chains for re-computation.
 rawkeys=['noise4_initial_noise','solver_budgets','executed_steps','noise4_seed','active','before','after','success_after']
 rawkeys += [f'{key}_n{n}' for n in [1,5,10] for key in ['actions_model','actions_env','x_chain','z_endpoint','timesteps','final_time']]
 for q in sorted(set(c['query'])):
  with np.load(d/f'query_{q:03d}.npz')as z:
   raw={k:(z[k][c['slot']:c['slot']+1] if z[k].ndim and z[k].shape[0]==len(read(d/'episodes.json')) else z[k])for k in rawkeys}
  np.savez_compressed(rawdir/f'query_{q:03d}.npz',**raw)
  shutil.copy2(d/f'q{q:03d}_s{c["slot"]:02d}_controller.npz',rawdir/f'q{q:03d}_controller.npz')
  with np.load(d/f'query_{q:03d}_obs.npz')as z:
   np.savez_compressed(rawdir/f'query_{q:03d}_obs.npz',**{k:z[k][c['slot']:c['slot']+1] for k in z.files})
 shutil.copy2(d/f'episode_{c["slot"]:02d}_control.jsonl',rawdir/'control.jsonl')
 print(json.dumps({'selected':c['id'],'task':c['task'],'slot':c['slot'],'success':c['success']}),flush=True)
manifest={'tasks':len(tasks),'selected':selected,'coverage':coverage,'episodes':sum(x['episodes']for x in coverage.values()),'successes':sum(x['successes']for x in coverage.values()),'errors':errors,'missing_tasks':sorted({p.stem.removesuffix('_b32') for p in (R/'configs').glob('*_b32.json')}-set(tasks)),'all_trajectory_stats':allstats,'source':str(R),'main_signal':'u_1_10','n1_dv':'not defined; missing, not zero'}
write(O/'manifest.json',manifest)
# Software H264 only: existing accepted encoder, original videos remain untouched.
def video(c):
 dest=O/c['id']/'episode-h264.mp4'
 if dest.exists():return
 subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(Path(c['remote_dir'])/c['video']),'-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(dest)],check=True,timeout=180,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
with concurrent.futures.ThreadPoolExecutor(3)as pool:list(pool.map(video,selected))
write(O/'export-complete.json',{'tasks':len(tasks),'selected':len(selected),'episodes':manifest['episodes'],'successes':manifest['successes'],'missing_tasks':manifest['missing_tasks'],'errors':errors})
print(json.dumps(read(O/'export-complete.json')),flush=True)
