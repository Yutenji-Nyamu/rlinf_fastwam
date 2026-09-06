#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,hashlib,json,os,re,subprocess
from pathlib import Path
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
assert os.getuid()==1003
tz=datetime.timezone(datetime.timedelta(hours=8))
def cmd(args,timeout=20):
    try:
        p=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
        return {'rc':p.returncode,'out':p.stdout,'err':p.stderr}
    except subprocess.TimeoutExpired:return {'rc':'timeout'}
    except OSError as e:return {'rc':'unavailable','err':str(e)}
def txt(p):
    try:return Path(p).read_text(errors='replace')
    except OSError:return None
base=Path('/data/chenyiteng/results/rlinf-shenzhen');wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
runpaths={
 'fastwam':base/'fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3',
 'sidney':base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1',
 'pi0':base/'grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2'}
s={'time':datetime.datetime.now(tz).isoformat(),'identity':cmd(['id']),'runs':{},'sources':[],'git':{}}
for name,root in runpaths.items():
    rt=root/'runtime'; log=txt(rt/'driver.log') or '';lines=log.splitlines()
    ea=EventAccumulator(str(root/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
    scalars={tag:[{'step':p.step+1,'value':p.value,'wall_time':p.wall_time} for p in ea.Scalars(tag)] for tag in ea.Tags().get('scalars',[])}
    state={k:txt(rt/k) for k in ['wrapper.pid','started_at.txt','finished_at.txt','exit_code.txt']}
    state['wrapper_alive']=Path('/proc', (state['wrapper.pid'] or '-1').strip()).exists()
    d={'path':str(root),'config':yaml.safe_load((rt/'resolved.yaml').read_text()),'resolved_text':txt(rt/'resolved.yaml'), 'scalars':scalars,'state':state,
       'completed_steps':re.findall(r'Global Step:\s*(\d+)\s*/',log)[-1:],
       'log_mtime':datetime.datetime.fromtimestamp((rt/'driver.log').stat().st_mtime,tz).isoformat(),
       'phase':[x for x in lines if 'Generating Rollout Epochs:' in x][-4:],'log_head':lines[:100],'log_tail':lines[-60:],
       'trainable_logs':[l for l in lines if re.search('trainable|freez|parameter count',l,re.I)][:40],
       'errors':{k:log.count(k) for k in ['Fatal Python error','OIDN Error','CUDA out of memory','OutOfMemoryError','RuntimeError:','Traceback']}}
    hits=[i for i,l in enumerate(lines) if 'Fatal Python error' in l]
    d['first_fatal_raw']=lines[max(0,hits[0]-150):hits[0]+600] if hits else []
    d['checkpoint_files']=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size} for p in root.glob('*/checkpoints/global_step_*/actor/**/*') if p.is_file()]
    s['runs'][name]=d

common=['rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py','rlinf/algorithms/utils.py',
 'rlinf/runners/embodied_runner.py','rlinf/workers/actor/embodied_fsdp_actor_worker.py','rlinf/workers/actor/fsdp_actor_worker.py',
 'rlinf/hybrid_engines/fsdp/fsdp_model_manager.py','rlinf/hybrid_engines/fsdp/strategy/fsdp2.py','rlinf/hybrid_engines/fsdp/strategy/checkpoint.py',
 'rlinf/utils/utils.py','rlinf/utils/metric_utils.py','rlinf/utils/checkpoint.py','rlinf/envs/robotwin/robotwin_env.py','rlinf/workers/rollout/hf/huggingface_worker.py',
 'rlinf/hybrid_engines/fsdp/strategy/base.py','rlinf/hybrid_engines/fsdp/utils.py','rlinf/models/embodiment/openpi/__init__.py',
 'rlinf/workers/env/env_worker.py','rlinf/data/embodied_io_struct.py','rlinf/models/embodiment/openpi/openpi_action_model.py']
def source(root,rel,label):
    p=root/rel
    if p.is_file():
        raw=p.read_bytes();s['sources'].append({'label':label,'root':str(root),'rel':rel,'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace')})
for label,name in [('fastwam','fastwam-current-grpo'),('sidney','sidney-pi05-current-rlinf'),('pi0','pi0-dvac-grpo-current')]:
    root=wt/name
    s['git'][label]={'path':str(root),'head':cmd(['git','--no-optional-locks','-C',str(root),'rev-parse','HEAD']), 'dirty':cmd(['git','--no-optional-locks','-C',str(root),'status','--porcelain'])}
    paths=set(common)
    if label=='fastwam': paths.update(str(p.relative_to(root)) for p in (root/'rlinf/models/embodiment/fastwam').glob('*.py'))
    paths.update(str(p.relative_to(root)) for p in (root/'rlinf/hybrid_engines/fsdp').rglob('*scheduler*.py'))
    for rel in sorted(paths):source(root,rel,label)
    s['git'][label]['scheduler_search']=[{'path':str(p.relative_to(root)),'line':i,'text':l} for tree in [root/'rlinf/hybrid_engines/fsdp',root/'rlinf/workers/actor'] for p in tree.rglob('*.py') for i,l in enumerate((txt(p) or '').splitlines(),1) if re.search('get_scheduler|build_lr_scheduler|lr_scheduler.*=|lr_warmup|num_training_steps',l)]
robot=wt/'robotwin-clean-oidn-off-20260904'
s['git']['robotwin']={'head':cmd(['git','--no-optional-locks','-C',str(robot),'rev-parse','HEAD']),'dirty':cmd(['git','--no-optional-locks','-C',str(robot),'status','--porcelain'])}
for rel in ['robotwin/envs/vector_env.py','envs/_base_task.py','envs/utils/camera.py','envs/move_stapler_pad.py','envs/move_pillbottle_pad.py','task_config/_eval_step_limit.yml']:source(robot,rel,'robotwin')
build=Path('/home/chenyiteng/builds/fastwam-scene-fence-20260904')
s['build_sources']=cmd(['find',str(build),'-maxdepth','3','-type','f','-name','*.cpp'])
s['core_pattern']=txt('/proc/sys/kernel/core_pattern');s['coredumps']=cmd(['coredumpctl','--no-pager','list','--since','2026-09-05 04:40:00','--until','2026-09-05 04:55:00'])
s['ray_error_files']=[]
for root in [Path('/tmp/ray/session_latest/logs'),Path('/data/chenyiteng/ray/session_latest/logs')]:
    if root.is_dir():
        for f in root.glob('*1569541*.err'):
            lines=(txt(f) or '').splitlines();hits=[i for i,l in enumerate(lines) if 'Fatal Python error' in l]
            s['ray_error_files'].append({'path':str(f),'fatal_raw':lines[max(0,hits[0]-100):hits[0]+600] if hits else lines[-40:]})
venv=Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages')
s['native_packages']={}
for pat in ['sapien-*.dist-info/METADATA','pybind11-*.dist-info/METADATA']:
    for f in venv.glob(pat):s['native_packages'][str(f)]=[l for l in (txt(f) or '').splitlines() if l.startswith(('Name:','Version:'))]
s['native_paths']=[str(p) for p in (venv/'sapien').rglob('*.so')]
s['health']={'gpu':cmd(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']), 'meminfo':txt('/proc/meminfo'),'disk':cmd(['df','-B1','/data']), 'processes':cmd(['ps','-p','3176215,321933,322685,1568973','-o','user:16,pid,etime,stat,rss,comm'])}
s['end_time']=datetime.datetime.now(tz).isoformat()
import base64,zlib
print(base64.b64encode(zlib.compress(json.dumps(s,ensure_ascii=False).encode(),9)).decode())
PY
