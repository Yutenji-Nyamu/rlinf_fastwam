import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json,sys,tarfile,argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--followup',required=True);p.add_argument('--output',required=True);args=p.parse_args();D=Path(args.data);D2=Path(args.followup);O=Path(args.output);O.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text());rows=[]
def frame(name,j,rgb,private=False):
 path=Path('private' if private else 'public')/'frames'/f'{name}_{j}.png';(O/path).parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(O/path);return str(path)
def finish(row):
 assert len(row['x'])==len(row['entropy'])==len(row['query']);assert np.isfinite(row['entropy']).all();assert np.all(np.diff(row['x'])>0);rows.append(row)
# Native RT2 projection and recorded video frame map are approximate per-arm alignment.
ep=next(e for e in read(D/'existing/index.json') if e['task']=='turn_switch');base=D/'existing'/ep['id'];p=read(base/'series.json')['positions'];fm=read(base/'frame_map.json')['frames'];video=next(Path(ep['source']).rglob(ep['video']));cap=cv2.VideoCapture(str(video));fr=[]
for j,i in enumerate([int(len(p)*.08),int(len(p)*.5),int(len(p)*.92)]):
 point=p[i];valid=[f for f in fm if f.get('query')==point['query']];f=min(valid,key=lambda f:abs(max(f.get('left_h_projection',-1),f.get('right_h_projection',-1))-point['h']))
 cap.set(cv2.CAP_PROP_POS_FRAMES,f['frame']);ok,bgr=cap.read();assert ok
 fr.append({'path':frame('signal_rt2',j,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),'label':f"action {point['action']}",'x':point['action'],'video_frame':f['frame'],'query':point['query'],'h':point['h'],'alignment':'nearest native per-arm projection'})
cap.release();chunks=[]
for q in sorted({x['query'] for x in p}):
 pp=[x for x in p if x['query']==q];chunks.append({'start':pp[0]['action'],'end':pp[-1]['action']+1})
finish({'id':'signal_rt2','visibility':'public','title':'RT2 | turn switch | success','xlabel':'Executed action index (native projection)','x':[v['action'] for v in p],'entropy':[v['entropy4'] for v in p],'query':[v['query'] for v in p],'chunks':chunks,'frames':fr,'protocol':'Original K4, normalized D14, diagonal Gaussian entropy, ddof=1, eps=1e-6; no smoothing','episode':ep['id']})
for task in ['put_bottles_into_dustbin','pour_liquid_into_cup']:
 folder=D2/'dojo-dense'/task;qs=read(folder/'queries.json');eps=read(folder/'episodes.json')
 for slot,e in enumerate(eps):
  seg=[]
  for q in qs:
   z=np.load(folder/q['path']);where=np.flatnonzero(z['slots']==slot)
   if len(where):
    i=where[0];start=int(z['before'][i]);n=int(z['after'][i])-start
    seg.append({'query':q['query'],'start':start,'n':n,'entropy':z['entropy'][i,:n].tolist()})
  if not seg:continue
  xx=[a for s in seg for a in range(s['start'],s['start']+s['n'])];assert np.all(np.diff(xx)==1),(task,slot)
  name=f'signal_dojo_{task}_{slot:02d}';fr=[]
  for j,idx in enumerate([0,len(seg)//2,len(seg)-1]):
   s=seg[idx];z=np.load(D/'dojo'/task/'observations'/f"q{s['query']:03d}.npz")
   fr.append({'path':frame(name,j,z['main_images'][slot]),'label':f"action {s['start']}",'x':s['start'],'query':s['query'],'alignment':'saved observation before action chunk'})
  finish({'id':name,'visibility':'public','title':f"Dojo | {'put bottles into dustbin' if task.startswith('put') else 'pour liquid into cup'} | {'success' if e['success'] else 'failure'} | episode {slot}",'xlabel':'Executed action index','x':xx,'entropy':[v for s in seg for v in s['entropy']],'query':[s['query'] for s in seg for _ in range(s['n'])],'chunks':[{'start':s['start'],'end':s['start']+s['n']} for s in seg],'frames':fr,'protocol':'K4 independent offline resampling at every reached saved observation. Concatenated actual executed prefixes; no interpolation or smoothing. Normalized D14.','episode':slot,'task':task,'success':e['success']})
idx=read(D/'geowam/index.json')
for episode in dict.fromkeys(r['episode'] for r in idx):
 rr=sorted([r for r in idx if r['episode']==episode],key=lambda r:r['observation_time_s']);name='signal_geowam_'+episode;fr=[];xx=[];hh=[];qq=[];chunks=[];gap=False
 for r in rr:
  h=np.load(D/'geowam'/r['directory']/'actions.npz')['entropy'];t=r['observation_time_s'];x=t+np.arange(len(h))/r['hz']
  if xx and abs(x[0]-xx[-1]-1/r['hz'])>.002:gap=True
  xx.extend(x.tolist());hh.extend(h.tolist());qq.extend([r['query']]*len(h));chunks.append({'start':float(x[0]),'end':float(x[-1]+1/r['hz'])})
 for j,i in enumerate([0,len(rr)//2,len(rr)-1]):
  r=rr[i];z=np.load(D/'geowam'/r['directory']/'observation.npz')
  fr.append({'path':frame(name,j,z['head'],True),'label':f"recorded {r['observation_time_s']:.1f}s",'x':r['observation_time_s'],'directory':r['directory'],'alignment':'recorded input frame; predicted future actions were not executed'})
 # Keep the contiguous selected example; discontinuous episodes are indexed without drawing fabricated bridges.
 if gap:continue
 finish({'id':name,'visibility':'private','title':f"GeoWAM | {rr[0]['task'].replace('_',' ')} | offline predictions",'xlabel':'Recorded time + predicted action offset (s)','x':xx,'entropy':hh,'query':qq,'chunks':chunks,'frames':fr,'protocol':'K4, H32, 10Hz. Recorded observations; predicted action sequence was not executed. All native stochastic noises vary. No smoothing.','episode':episode,'task':rr[0]['task'],'split':rr[0]['split']})
(O/'signals.json').write_text(json.dumps(rows,ensure_ascii=False))
sys.path.insert(0,str(Path(__file__).parent));from render_signals import render
render(O)
print(json.dumps({'figures':len(rows),'public':sum(r['visibility']=='public' for r in rows),'private':sum(r['visibility']=='private' for r in rows),'points':{r['id']:len(r['x']) for r in rows}}))
