import json,os,re,time
from pathlib import Path
root=Path('/data/chenyiteng/results');out=[]
for folder,ds,fs in os.walk(root):
 ds[:]=[d for d in ds if d not in ['.git','replay','replay_buffer','online','videos','video','images','frames','dataset','datasets']]
 p=Path(folder)
 if 'full_weights.pt' in fs and '/global_step_' in str(p):
  cp=next(a for a in p.parents if a.name.startswith('global_step_'));run=cp.parent.parent;cfg=None
  for a in [run,run.parent,run.parent.parent]:
   for f in [a/'runtime/resolved.yaml',a/'tensorboard/config.yaml',a/'config.yaml']:
    if f.is_file():cfg=f;break
   if cfg:break
  item={'checkpoint':str(cp),'weights':str(p/'full_weights.pt'),'bytes':(p/'full_weights.pt').stat().st_size,'step':int(cp.name.split('_')[-1]),'run':str(run),'config':str(cfg) if cfg else None}
  if cfg:item['config_evidence']=[l.strip() for l in cfg.read_text(errors='replace').splitlines() if any(k in l for k in ['task_name:','model_path:','model_type:','num_action_chunks:','num_steps:','denoising_steps:','model_name:'])]
  out.append(item)
 if 'checkpoint-latest.pt' in fs and p.name=='run':
  cfg=p.parent/'inputs.json';out.append({'checkpoint':str(p/'checkpoint-latest.pt'),'bytes':(p/'checkpoint-latest.pt').stat().st_size,'run':str(p.parent),'config':json.loads(cfg.read_text()) if cfg.exists() else None,'receipt':json.loads((p/'checkpoint.json').read_text()) if (p/'checkpoint.json').exists() else None})
# Keep only the most recent checkpoint per exact run; no tensor loading or compatibility claim.
latest={}
for x in out:
 if x['run'] not in latest or x.get('step',0)>latest[x['run']].get('step',0):latest[x['run']]=x
print(json.dumps({'time':time.time(),'candidates':list(latest.values())}))
