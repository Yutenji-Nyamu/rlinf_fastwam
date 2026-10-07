import base64,hashlib,io,json,os,socket,sys
from pathlib import Path
assert os.getuid()==20001 and socket.gethostname()=='h100-gpu02' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from PIL import Image
torch.set_num_threads(1)
T=Path('/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-turn-switch-repair-20261002');S=T.parent/'parallel-trial-20261006/source'
sys.path.insert(0,str(S))
from rlinf.algorithms.expo_ft.formal_replay import _digest
idx=json.loads((T/'replay/index.json').read_text())['online_entries'];output=[]
for number in (194,196):
 e=idx[number-1];p=T/'replay'/e['path'];data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==e['pin']['sha256']
 v=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False);del data
 m=json.loads((T/'replay'/e['manifest_path']).read_text())
 assert len(v['observations'])==e['frames'] and [_digest(o) for o in v['observations']]==m['frame_sha256'] and _digest(v['final_obs'])==m['final_obs_sha256']
 frames=[];n=e['frames']
 for step in (0,n//3,2*n//3,n):
  o=v['final_obs'] if step==n else v['observations'][step]
  main=o['main_images'][0].cpu()
  if main.dtype!=torch.uint8:main=(main*255).round().clamp(0,255).to(torch.uint8)
  image=Image.fromarray(main.numpy());image.thumbnail((420,320));buf=io.BytesIO();image.save(buf,format='PNG')
  frames.append({'step':step,'png':base64.b64encode(buf.getvalue()).decode()})
 output.append({'episode':number,'success':e['success'],'actions':n,'prompt':v['observations'][0]['task_descriptions'][0],
                'reward_sum':float(v['rewards'].sum()),'full_sha_and_frames_verified':True,'frames':frames})
 del v
print(json.dumps(output))
