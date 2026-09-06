#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,json,re,subprocess
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
rt=root/'runtime';newrt=root/'runtime-resume100-to200'
tz=datetime.timezone(datetime.timedelta(hours=8))
state={k:(rt/k).read_text().strip() if (rt/k).exists() else None for k in ['started_at.txt','finished_at.txt','exit_code.txt','wrapper.pid']}
log=(rt/'driver.log').read_text(errors='replace')
d={'time':datetime.datetime.now(tz).isoformat(),'run':str(root),'state':state,'completed_steps':re.findall(r'Global Step:\s*(\d+)\s*/',log)[-1:],'wrapper_alive':Path('/proc',state['wrapper.pid']).exists(),'phase':[l for l in log.splitlines() if 'Generating Rollout Epochs:' in l][-3:],'driver_tail':log.splitlines()[-160:],'errors':{k:log.count(k) for k in ['Fatal Python error','CUDA out of memory','RuntimeError:','Traceback (most recent call last):']}}
ea=EventAccumulator(str(root/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
d['scalars']={k:[{'step':x.step+1,'value':x.value,'wall_time':x.wall_time} for x in ea.Scalars(k) if x.step<100] for k in ea.Tags()['scalars']}
d['checkpoint_files']={'100':[{'path':str(p),'bytes':p.stat().st_size,'mtime':datetime.datetime.fromtimestamp(p.stat().st_mtime,tz).isoformat()} for p in root.glob('*/checkpoints/global_step_100/**/*') if p.is_file()]}
d['files']={alias:(rt/name).read_text(errors='replace') for alias,name in [('resolved.yaml','resolved.yaml'),('command.sh','command.txt'),('runtime/resources.csv','resource.csv'),('source_lock.txt','source-head.txt'),('launch_manifest.txt','launch_manifest.txt')] if (rt/name).is_file()}
d['gpu45']=subprocess.run(['nvidia-smi','-i','4,5','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
d['resume_state']={k:(newrt/k).read_text().strip() if (newrt/k).exists() else None for k in ['started_at.txt','finished_at.txt','exit_code.txt','wrapper.pid']}
if (newrt/'driver.log').is_file():d['resume_tail']=(newrt/'driver.log').read_text(errors='replace').splitlines()[-50:]
if state['exit_code.txt']=='0' and not (newrt/'launch_attempt.json').exists():
    import ray
    ray.init(address='172.17.0.1:6389',namespace='codex_pi05_finish_readonly',logging_level='ERROR')
    d['old_rlinf_named_actors']=[r['name'] for r in ray.util.list_named_actors(all_namespaces=True) if r.get('namespace')=='RLinf']
    ray.shutdown()
print(json.dumps(d))
PY
