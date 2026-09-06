set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import datetime,hashlib,json,subprocess
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=35);return {'rc':p.returncode,'out':p.stdout,'err':p.stderr}
result={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'trees':{},'timings':{}}
for name in ('pi0-online-bc','pi0-online-bc-dvac','sidney-pi05-current-rlinf'):
 root=base/name
 paths=['rlinf/models/embodiment/openpi/openpi_action_model.py','rlinf/models/embodiment/openpi/__init__.py','rlinf/models/embodiment/openpi/dataconfig/aloha.py','rlinf/models/embodiment/openpi/dataconfig/__init__.py','rlinf/workers/actor/fsdp_online_bc_policy_worker.py','rlinf/workers/actor/fsdp_dagger_policy_worker.py','rlinf/data/online_bc.py','examples/embodiment/config/model/pi0.yaml','examples/embodiment/config/model/pi0_5.yaml','examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml','examples/embodiment/config/robotwin_move_stapler_pad_grpo_openpi_pi05_sidney.yaml','examples/sft/config/robotwin_sft_openpi_pi05.yaml']
 files={p:{'sha256':hashlib.sha256((root/p).read_bytes()).hexdigest(),'text':(root/p).read_text()} for p in paths if (root/p).is_file()}
 result['trees'][name]={'head':cmd(['git','-C',str(root),'rev-parse','HEAD']),'dirty':cmd(['git','-C',str(root),'status','--porcelain']),'files':files,'sidney_paths':cmd(['git','-C',str(root),'log','--all','--format=%H %s','--grep=Sidney','-8']),'dataconfig_paths':cmd(['git','-C',str(root),'grep','-l','pi05_sidney_robotwin','--','rlinf'])}
import openpi.models_pytorch.pi0_pytorch as native
native_path=Path(native.__file__)
result['native']={'path':str(native_path),'sha256':hashlib.sha256(native_path.read_bytes()).hexdigest(),'text':native_path.read_text()}
gemma=native_path.parent/'gemma_pytorch.py'
if gemma.exists():result['gemma']={'path':str(gemma),'sha256':hashlib.sha256(gemma.read_bytes()).hexdigest(),'text':gemma.read_text()}
bc=Path('/data/chenyiteng/results/rlinf-shenzhen/online-bc')
runs={'formal':'pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1','smoke_v7':'pi0-adjust-bottle-smoke32x1-b1024-u10-eval16x2-gpu6-20260905-v7','smoke_v8':'pi0-adjust-bottle-smoke32x1-b1024-u10-eval8x4-gpu6-20260905-v8','dvac_smoke':'pi0-bc-dvac32x1-b1024-u10-gpu7-smoke2-20260905-v1'}
for name,folder in runs.items():
 root=bc/folder;ea=EventAccumulator(str(root/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
 scalars={t:[{'step':x.step+1,'value':x.value} for x in ea.Scalars(t)] for t in ea.Tags()['scalars'] if any(k in t for k in ('time/','success','loss','grad_norm','replay_buffer/'))}
 configs={str(p.relative_to(root)):p.read_text() for p in (root/'runtime').glob('*.yaml')}
 if (root/'resolved.yaml').exists():configs['resolved.yaml']=(root/'resolved.yaml').read_text()
 logs={str(p.relative_to(root)):p.read_text(errors='replace')[-40000:] for p in (root/'driver.log',root/'runtime/driver.log') if p.exists()}
 result['timings'][name]={'path':str(root),'scalars':scalars,'configs':configs,'log_tail':logs}
print(json.dumps(result))
PY
