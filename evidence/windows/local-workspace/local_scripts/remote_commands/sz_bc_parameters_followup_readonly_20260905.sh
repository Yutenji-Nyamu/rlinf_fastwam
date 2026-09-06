set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,json,re,subprocess
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
run=base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1'
rt=run/'runtime-resume100-to200'; log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',(rt/'driver.log').read_text(errors='replace'))
ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
tags=('env/success_once','eval/success_once','time/step','train/actor/grad_norm','train/actor/policy_loss','train/actor/approx_kl','train/actor/clip_fraction')
r={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
   'scalars':{tag:[{'step:p':p.step,'step':p.step+1,'value':p.value,'wall_time':p.wall_time} for p in ea.Scalars(tag)] for tag in tags},
   'log_step':re.findall(r'Global Step:\s*(\d+)\s*/',log)[-1:],
   'phase':[l for l in log.splitlines() if 'Generating Rollout Epochs:' in l][-2:],
   'last_table':log[log.rfind('Global Step:'):][-14000:],
   'errors':{k:log.count(k) for k in ('Fatal Python error','CUDA out of memory','OutOfMemoryError','Traceback (most recent call last):','RuntimeError:')},
   'wrapper_alive':Path('/proc',(rt/'wrapper.pid').read_text().strip()).exists(),
   'exit':(rt/'exit_code.txt').read_text().strip() if (rt/'exit_code.txt').exists() else None}
formal=base/'online-bc/pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1'
r['pi0_formal_config_paths']=[str(p) for p in formal.glob('*/*resolved*')]+[str(p) for p in formal.glob('*resolved*')]
r['pi0_formal_config']={str(p):p.read_text() for p in formal.glob('*/*resolved*') if p.is_file()}
tree=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc')
q=subprocess.run(['git','-C',str(tree),'grep','-n','-E','def save_checkpoint|full_weights.pt','--','rlinf/hybrid_engines/fsdp'],capture_output=True,text=True)
r['save_source_locations']=q.stdout
sources={}
for name in set(line.split(':',1)[0] for line in q.stdout.splitlines()):
    text=(tree/name).read_text();ls=text.splitlines();ranges=[]
    for i,s in enumerate(ls):
        if 'def save_checkpoint(' in s or 'full_weights.pt' in s:ranges.append({'line':i+1,'text':'\n'.join(ls[max(0,i-8):i+105])})
    sources[name]=ranges
r['save_source']=sources
print(json.dumps(r))
PY
