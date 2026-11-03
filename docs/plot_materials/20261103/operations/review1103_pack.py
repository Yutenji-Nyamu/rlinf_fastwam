import json,os,shutil,time,tarfile
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
r=json.loads(Path('/home/chenyiteng/deployment-20261102/review1103-private.json').read_text())
g=r['grpo']['plan'];root=Path(g['run']);ea=EventAccumulator(str(root/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
scalars={tag:[{'step':x.step,'wall_time':x.wall_time,'value':x.value}for x in ea.Scalars(tag)]for tag in ea.Tags()['scalars']}
cp=Path(g['checkpoint_root']);checkpoints=[]
for d in cp.glob('global_step_*'):
 files=[f for f in d.rglob('*')if f.is_file()];checkpoints.append({'step':int(d.name.rsplit('_',1)[-1]),'files':len(files),'gib':sum(f.stat().st_size for f in files)/2**30,'full_weights':(d/'actor/model_state_dict/full_weights.pt').is_file()})
out=Path('/data/chenyiteng/exports/plot-backup-20261103');out.mkdir(exist_ok=True)
(out/'grpo-scalars.json').write_text(json.dumps(scalars));(out/'grpo-checkpoints.json').write_text(json.dumps(checkpoints))
roots={k:Path(x['output'])for k,x in r['runs'].items()if k!='fql-fix'};roots['clean-grpo']=root
# Same output may contain two methods; store it once.
seen=set();rows=[];skip=[]
for name,base in roots.items():
 real=base.resolve()
 if real in seen:continue
 seen.add(real)
 for dp,dn,fn in os.walk(base):
  dn[:]=[d for d in dn if d not in ['checkpoints','video','videos','replay','wandb','__pycache__']]
  for n in fn:
   f=Path(dp)/n;rel=f.relative_to(base)
   if f.suffix.lower()not in ['.json','.jsonl','.csv','.yaml','.yml','.txt','.log'] and not n.startswith('events.out.tfevents'):continue
   if any(word in n.lower()for word in ['environment','credential','secret']):continue
   size=f.stat().st_size
   if size>32*2**20:skip.append({'run':name,'path':str(rel),'bytes':size});continue
   dest=out/'runs'/name/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
   rows.append({'run':name,'path':str(rel),'bytes':dest.stat().st_size})
manifest={'time':time.time(),'files':rows,'large_excluded':skip,'excluded':'weights, replay, full video, images, environments, credentials; known older figure archives indexed separately'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz')as t:
 for f in sorted(out.rglob('*')):
  if f.is_file():t.add(f,arcname=str(f.relative_to(out)))
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(rows),'raw_mib':sum(x['bytes']for x in rows)/2**20,'skipped':skip,'grpo_checkpoints':checkpoints,'grpo_eval_tags':{k:v for k,v in scalars.items()if 'eval' in k and ('success' in k or 'return' in k)}}))
