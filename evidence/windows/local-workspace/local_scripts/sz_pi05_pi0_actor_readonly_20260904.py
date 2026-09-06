from pathlib import Path
import datetime, hashlib, json, re, subprocess
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def emit(k,x): print(k,json.dumps(x,ensure_ascii=False),flush=True)
def source(p):
    raw=p.read_bytes();emit('SOURCE_JSON',{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace')})
emit('TIME_JSON',str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))))
root=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
for name in ['pi0-dvac-grpo-current','sidney-pi05-current-rlinf']:
    wt=root/name
    for folder in ['rlinf/workers/actor','rlinf/workers/rollout']:
        for p in (wt/folder).glob('**/*.py'):
            text=p.read_text()
            if 'class EmbodiedFSDPActor' in text or 'class MultiStepRolloutWorker' in text:
                source(p)
    for rel in ['rlinf/envs/robotwin/seed_utils.py','rlinf/algorithms/utils.py']:
        p=wt/rel
        if p.exists():source(p)
    for rel in ['rlinf/models/embodiment/openpi/openpi_action_model.py','rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py']:
        emit('BASELINE_DIFF_JSON',{'wt':name,'rel':rel,'out':subprocess.run(['git','--no-optional-locks','-C',str(wt),'diff','0e28ac6f09f821ea12e7d54eba7118ce0000ca86','HEAD','--',rel],capture_output=True,text=True).stdout})
for name in ['rlinf-native/sidney-pi05-robotwin-e49e2ab','rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50']:
    model=Path('/data/chenyiteng/models')/name
    for p in model.glob('**/norm_stats.json'):source(p)
run=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
log=(run/'runtime/driver.log').read_text(errors='replace')
emit('LATEST_SCALARS_JSON',{'time':str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))),
    'progress':re.findall(r'Global Step:\s*\d+/\d+',log)[-1:],
    'scalars':{tag:[{'step':p.step+1,'value':p.value,'wall_time':p.wall_time} for p in ea.Scalars(tag)] for tag in ea.Tags()['scalars']},
    'errors':{x:log.lower().count(x.lower()) for x in ['Fatal Python error','OIDN Error','CUDA out of memory','Traceback','RuntimeError:']}})
