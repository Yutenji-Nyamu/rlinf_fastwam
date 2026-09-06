set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' GIT_OPTIONAL_LOCKS=0
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,hashlib,json,re,subprocess
from pathlib import Path
base=Path('/data/chenyiteng');trees=base/'projects/rlinf-shenzhen/worktrees'
paths={'bc':base/'results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1',
'dvac':base/'results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1'}
def source(p):
    if not p.is_file():return {'path':str(p),'exists':False}
    t=p.read_text();return {'path':str(p),'sha256':hashlib.sha256(t.encode()).hexdigest(),'text':t}
d={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'runs':{},'source':{}}
for name,run in paths.items():
    d['runs'][name]={'configs':{str(p.relative_to(run)):source(p) for p in run.glob('runtime/*') if p.name in ['resolved.yaml','wrapper.sh','environment.sh','source-head.txt']},
      'seed_log':[l for l in (run/'driver.log').read_text(errors='replace').splitlines() if re.search('seed|reset_state|manual_seed|determin',l,re.I)][:80]}
for name,tree in [('bc',trees/'pi05-online-bc'),('dvac',trees/'pi05-online-bc-dvac')]:
    files=['rlinf/envs/robotwin/robotwin_env.py','rlinf/envs/robotwin/seed_utils.py','rlinf/workers/env/env_worker.py',
      'rlinf/workers/rollout/multi_step_rollout_worker.py','rlinf/models/embodiment/openpi/openpi_model.py',
      'rlinf/data/online_bc.py','rlinf/workers/actor/fsdp_online_bc_policy_worker.py','rlinf/runners/embodied_runner.py']
    d['source'][name]={f:source(tree/f) for f in files}
    q=subprocess.run(['git','-C',str(tree),'grep','-n','-E','set_seed|manual_seed|seed_everything|fork_rng|randn|sample_noise','--','rlinf/workers/rollout','rlinf/models/embodiment/openpi','rlinf/utils'],capture_output=True,text=True)
    d['source'][name]['rng_search']=q.stdout
    for f in ['rlinf/envs/robotwin/seeds/train.json','rlinf/envs/robotwin/seeds/eval_sidney_fixed32.json']:
        d['source'][name][f]=source(tree/f)
rt=base/'projects/rlinf-shenzhen/RoboTwin-RLinf-support'
for f in ['robotwin/envs/vector_env.py','robotwin/envs/_base_task.py','envs/vector_env.py','envs/_base_task.py']:
    if (rt/f).is_file():d['source']['robotwin/'+f]=source(rt/f)
d['robotwin_rng_search']=subprocess.run(['git','-C',str(rt),'grep','-n','-E','def set_seed|np.random.seed|random.seed|default_rng|manual_seed','--','robotwin/envs','envs'],capture_output=True,text=True).stdout
print(json.dumps(d,ensure_ascii=False))
PY
