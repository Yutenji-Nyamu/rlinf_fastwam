set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' GIT_OPTIONAL_LOCKS=0
nice -n 19 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime,difflib,hashlib,json,os,subprocess
from pathlib import Path
import yaml
base=Path('/data/chenyiteng');trees=base/'projects/rlinf-shenzhen/worktrees'
rs=base/'results/rlinf-shenzhen/online-bc'
runs={'bc':rs/'pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1',
      'dvac':rs/'pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1'}
roots={'bc':trees/'pi05-online-bc','dvac':trees/'pi05-online-bc-dvac'}
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=40);return {'rc':p.returncode,'out':p.stdout,'err':p.stderr}
def src(p):
 if not p.is_file():return {'exists':False,'path':str(p)}
 b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
files=['rlinf/algorithms/online_bc_dvac.py','rlinf/data/online_bc.py','rlinf/workers/actor/fsdp_online_bc_policy_worker.py',
'rlinf/workers/actor/fsdp_dagger_policy_worker.py','rlinf/workers/rollout/hf/huggingface_worker.py',
'rlinf/workers/env/env_worker.py','rlinf/models/embodiment/openpi/openpi_action_model.py',
'rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py','rlinf/runners/embodied_runner.py',
'examples/embodiment/config/bc_dvac/default.yaml','examples/embodiment/config/bc_dvac/bounded_half.yaml',
'tests/unit_tests/test_online_bc_dvac.py','tests/unit_tests/test_pi05_online_bc_dvac.py']
d={'time':datetime.datetime.now().astimezone().isoformat(),'runs':{},'files':{},'diffs':{}}
configs={}
for n,r in runs.items():
 d['runs'][n]={'runtime':{p.name:src(p) for p in (r/'runtime').iterdir() if p.name in ['resolved.yaml','wrapper.sh','environment.sh','source-head.txt']},
 'git_head':cmd(['git','-C',str(roots[n]),'rev-parse','HEAD']),
 'git_status':cmd(['git','-C',str(roots[n]),'status','--porcelain']),
 'unstable_retry_lines':[l for l in (r/'driver.log').read_text(errors='replace').splitlines() if any(t in l for t in ['UnStable','Unstable','trial_seed','retry seed','resampl'])]}
 configs[n]=yaml.safe_load((r/'runtime/resolved.yaml').read_text())
 d['files'][n]={f:src(roots[n]/f) for f in files}
def flat(x,p=''):
 if isinstance(x,dict):
  out={}
  for k,v in x.items():out.update(flat(v,p+'.'+k if p else k))
  return out
 return {p:x}
a,b=flat(configs['bc']),flat(configs['dvac'])
d['resolved_diff']=[{'field':k,'bc':a.get(k),'dvac':b.get(k)} for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)]
for f in files:
 aa=d['files']['bc'][f].get('text','');bb=d['files']['dvac'][f].get('text','')
 if aa!=bb:d['diffs'][f]=''.join(difflib.unified_diff(aa.splitlines(True),bb.splitlines(True),fromfile='bc/'+f,tofile='dvac/'+f))
d['full_code_diff_stat']=cmd(['git','-C',str(roots['bc']),'diff','--stat',d['runs']['bc']['git_head']['out'].strip(),d['runs']['dvac']['git_head']['out'].strip(),'--','rlinf','examples','tests'])
d['all_source_names']=cmd(['git','-C',str(roots['bc']),'diff','--name-status',d['runs']['bc']['git_head']['out'].strip(),d['runs']['dvac']['git_head']['out'].strip(),'--','rlinf','examples','tests'])
print(json.dumps(d,ensure_ascii=False))
PY
