from pathlib import Path
import datetime, hashlib, json, subprocess
import openpi
def emit(kind, value): print(kind, json.dumps(value, ensure_ascii=False), flush=True)
def source(p):
    if p.is_file():
        raw=p.read_bytes();emit('SOURCE_JSON', {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace')})
emit('TIME_JSON', str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))))
root=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
for name in ['pi0-dvac-grpo-current','sidney-pi05-current-rlinf']:
    for rel in ['rlinf/models/embodiment/openpi/dataconfig/__init__.py','rlinf/workers/rollout/hf/multi_step_rollout_worker.py']:
        source(root/name/rel)
    p=root/name
    for args in [['log','-6','--format=%H %s'],['log','-5','--format=%H %s','--','rlinf/models/embodiment/openpi/openpi_action_model.py']]:
        emit('GIT_JSON',{'root':str(p),'args':args,'out':subprocess.run(['git','--no-optional-locks','-C',str(p),*args],capture_output=True,text=True).stdout})
src=Path(openpi.__file__).parent
for rel in ['models/pi0_config.py','models_pytorch/pi0_pytorch.py','models/tokenizer.py','transforms.py']:
    source(src/rel)
robot=Path('/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support')
for p in (robot/'robotwin').glob('**/*.py'):
    if p.name in ['wrapper.py','env.py','robotwin_env.py','base_env.py'] or 'reward' in p.name: source(p)
emit('ROBOTWIN_FILES_JSON',[str(p) for p in (robot/'robotwin').glob('**/*.py')])
results=Path('/data/chenyiteng/results/rlinf-shenzhen')
paths=[results/'grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2/runtime',
       results/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1/runtime']
for p in paths:
    for name in ['source_head.txt','source-head.txt','launch_manifest.txt','contract.json','stopped_by_user_for_action_adv.txt','command.txt']:
        source(p/name)
    if 'sidney' in str(p):
        log=(p/'driver.log').read_text(errors='replace')
        rows=log.splitlines(); hits=[i for i,l in enumerate(rows) if 'Global Step:' in l]
        emit('LATEST_LOG_JSON',{'path':str(p),'tail':rows[hits[-1]:][-100:] if hits else rows[-30:]})
for model in [Path('/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab'),Path('/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50')]:
    emit('MODEL_FILES_JSON',{'path':str(model),'names':[str(p.relative_to(model)) for p in model.glob('*')]})
    for rel in ['config.json','conversion_manifest.json','assets/physical-intelligence/robotwin/norm_stats.json']:
        source(model/rel)
