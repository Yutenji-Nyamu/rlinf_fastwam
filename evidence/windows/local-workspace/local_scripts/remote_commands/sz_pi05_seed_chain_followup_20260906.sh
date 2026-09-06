set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,hashlib,importlib.util,json,os,subprocess
from pathlib import Path
import torch,yaml
base=Path('/data/chenyiteng');trees=base/'projects/rlinf-shenzhen/worktrees';b=trees/'pi05-online-bc';d={'time':datetime.datetime.now().astimezone().isoformat(),'files':{},'seeds':{}}
def read(p):
    if not p.is_file():return {'path':str(p),'exists':False}
    raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace')}
def cmd(a):
    p=subprocess.run(a,capture_output=True,text=True,timeout=45,env={k:v for k,v in os.environ.items() if k.lower() not in ['http_proxy','https_proxy','all_proxy']})
    return {'rc':p.returncode,'out':p.stdout,'err':p.stderr}
for name,tree in [('bc',b),('dvac',trees/'pi05-online-bc-dvac')]:
    files=['rlinf/models/embodiment/openpi/openpi_action_model.py','rlinf/models/__init__.py','rlinf/utils/utils.py','rlinf/utils/initialize.py']
    q=cmd(['git','-C',str(tree),'grep','-l','-E','class MultiStepRolloutWorker|class HuggingFaceWorker|class Worker\(','--','rlinf'])
    files+=q['out'].splitlines()
    for f in files:d['files'][name+'/'+f]=read(tree/f)
    d['rng_callers_'+name]=cmd(['git','-C',str(tree),'grep','-n','-E','seed_everything|set_random_seed|set_seed\(|manual_seed','--','rlinf/workers','rlinf/scheduler','rlinf/hybrid_engines'])
    spec=importlib.util.spec_from_file_location('seed_partition_'+name,tree/'rlinf/envs/robotwin/seed_utils.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    d['seeds'][name]={}
    for mode,filename,n in [('train','train_seeds.json',32),('eval','eval_sidney_fixed32.json',8)]:
        p=tree/'rlinf/envs/robotwin/seeds'/filename;ss=json.loads(p.read_text())['move_pillbottle_pad']['success_seeds']
        ordered=m.partition_success_seeds(torch.as_tensor(ss),base_seed=0,seed_offset=0,total_num_processes=1,num_group=n).tolist()
        if mode=='eval':ordered=ordered[:32]
        d['seeds'][name][mode]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'count_raw':len(ss),'ordered_count':len(ordered),'ordered':ordered,'first_batches':[ordered[(i*n)%len(ordered):(i*n)%len(ordered)+n] for i in range(4)]}
rt=base/'projects/rlinf-shenzhen/RoboTwin-RLinf-support'
for f in ['description/utils/generate_episode_instructions.py','envs/move_pillbottle_pad.py']:
    d['files']['robotwin/'+f]=read(rt/f)
for p in Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin').glob('lib/python*/site-packages/openpi/models_pytorch/pi0_pytorch.py'):
    t=p.read_text();ls=t.splitlines();d['native_sample_noise']={'path':str(p),'sha256':hashlib.sha256(t.encode()).hexdigest(),'snippets':[{'line':i+1,'text':'\n'.join(ls[max(0,i-3):i+20])} for i,l in enumerate(ls) if 'def sample_noise' in l]}
d['robotwin_remote_retry']={'status':'Previous separate HTTP/1.1 read timed out at 45s; not retried again.'}
print(json.dumps(d,ensure_ascii=False))
PY
