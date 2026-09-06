#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf')
d={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'head':subprocess.check_output(['git','-C',str(wt),'rev-parse','HEAD'],text=True).strip(),'dirty':subprocess.check_output(['git','-C',str(wt),'status','--porcelain'],text=True),'files':{}}
targets=[root/'runtime'/n for n in ['command.txt','resolved.yaml','wrapper.sh','observer.sh','source-head.txt']]+[root/'tensorboard/config.yaml']
for rel in ['rlinf/utils/metric_logger.py','rlinf/runners/embodied_runner.py','rlinf/envs/wrappers/record_video.py','rlinf/envs/robotwin/robotwin_env.py','rlinf/workers/env/env_worker.py','examples/embodiment/train_embodied_agent.py']:
    targets.append(wt/rel)
for p in targets:
    if p.is_file():
        b=p.read_bytes();d['files'][str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode('utf-8',errors='replace')}
d['wrapper_pid']=(root/'runtime/wrapper.pid').read_text().strip()
try:d['wrapper_environ']=Path('/proc',d['wrapper_pid'],'environ').read_bytes().decode().split('\0')
except FileNotFoundError:d['wrapper_environ']=None
if d['wrapper_environ'] is not None:
    allow={'PATH','PYTHONPATH','RAY_ADDRESS','ROBOTWIN_PATH','ROBOT_PLATFORM','REPO_PATH','EMBODIED_PATH','RLINF_CODE_WORKING_DIR','OPENPI_DATA_HOME','MUJOCO_GL','PYOPENGL_PLATFORM','HYDRA_FULL_ERROR','PYTHONUNBUFFERED','PYTHONDONTWRITEBYTECODE','VIRTUAL_ENV'}
    d['wrapper_environ']=[v for v in d['wrapper_environ'] if v.split('=',1)[0] in allow]
d['checkpoint100']=[{'path':str(p),'bytes':p.stat().st_size} for p in root.glob('*/checkpoints/global_step_100/**/*') if p.is_file()]
d['video_examples']=[str(p) for p in (root/'video').rglob('*.mp4')][:8]
d['data_examples']=[str(p) for p in (root/'robotwin_data').rglob('*')][:16]
print(json.dumps(d))
PY
