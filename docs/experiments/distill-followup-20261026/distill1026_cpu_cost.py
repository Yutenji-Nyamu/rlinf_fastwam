import os,json,sys,time,io
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
R=Path('/data/chenyiteng/deployment-20261025/endpoint-multitask-c10-sz1-g45/source');sys.path.insert(0,str(R))
import torch,numpy as np,pyarrow.parquet as pq
from rlinf.algorithms.pi05_distillation.data import decode_records
torch.set_num_threads(1)
spec=json.loads((R.parent/'data-manifest.json').read_text());records=[];sources=[]
for item in [spec['tasks'][i]for i in [0,17,41]]:
 root=Path(item['path']);p=sorted(root.glob('data/*/episode_*.parquet'))[0]
 keys=['observation.state','action','task_index']+[f'observation.images.{k}'for k in ['cam_high','cam_left_wrist','cam_right_wrist']]
 t=pq.read_table(p,columns=keys);row={k:t[k].to_pylist()for k in keys};row['actions']=np.asarray(row.pop('action'),dtype=np.float32);row['states']=np.asarray(row.pop('observation.state'),dtype=np.float32)
 prompts={int(r['task_index']):r['task']for r in map(json.loads,(root/'meta/tasks.jsonl').read_text().splitlines())}
 inds=np.linspace(0,len(row['actions'])-50,64,dtype=int)
 records.extend((row,int(i),prompts[int(row['task_index'][i])])for i in inds);sources.append({'task':item['task'],'episode':str(p),'frames':len(row['actions'])})
t=time.perf_counter();obs,a=decode_records(records,50);elapsed=time.perf_counter()-t
venv=Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages');gemma=venv/'transformers/models/gemma/modeling_gemma.py';text=gemma.read_text();lines=text.splitlines();hits=[]
for i,l in enumerate(lines):
 if '_attn_implementation' in l or 'attention_interface' in l:hits.append({'line':i+1,'context':'\n'.join(lines[max(0,i-3):i+6])})
print(json.dumps({'scope':'CPU-only representative 192 windows, 3 tasks, one episode each; decode+stack only; not a full update profile','decode_stack_s':elapsed,'main_image_shape':list(obs['main_images'].shape),'total_rgb_mib':(obs['main_images'].numel()+obs['wrist_images'].numel())/2**20,'sources':sources,'cuda_initialized':torch.cuda.is_initialized(),'gemma_attention':hits}))
