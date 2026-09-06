set -eu
root=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root:/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
export REPO_PATH="$root" EMBODIED_PATH="$root/examples/embodiment"
export ASSETS_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export PI05_MODEL_PATH=/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab
export ONLINE_BC_RUN_DIR=/data/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1
export RAY_ADDRESS=172.17.0.1:6389 RLINF_CODE_WORKING_DIR="$root"
cd "$root"
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import datetime,json,os,subprocess
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from rlinf.config import validate_cfg
import ray
def cmd(*a):return subprocess.check_output(a,text=True,timeout=20).strip()
root=Path.cwd(); run=Path(os.environ['ONLINE_BC_RUN_DIR'])
smoke=run.parent/'pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1'
head=cmd('git','rev-parse','HEAD'); status=cmd('git','status','--porcelain')
assert head=='912bc6907d39a0eec1eb98a6d4c9358e69791924' and not status,(head,status)
assert cmd('git','branch','--show-current')=='codex/sz-pi05-online-bc'
assert not run.exists(),str(run)
assert not cmd('nvidia-smi','-i','6','--query-compute-apps=pid','--format=csv,noheader,nounits')
assert (smoke/'exit_code.txt').read_text().strip()=='0'
assert (Path(os.environ['PI05_MODEL_PATH'])/'model.safetensors').stat().st_size==9354045872
overrides=['+online_bc_model=pi05_sidney','runner.max_epochs=100','runner.val_check_interval=5','runner.save_interval=10','actor.optim.total_training_steps=1000','runner.logger.experiment_name=pi05-pillbottle-bc-u10-eval8x4-formal100-gpu6']
with initialize_config_dir(version_base='1.1',config_dir=str(root/'examples/embodiment/config')):
 cfg=compose(config_name='robotwin_adjust_bottle_online_bc_openpi',overrides=overrides)
 try:cfg=validate_cfg(cfg)
 finally:ray.shutdown()
def leaves(v,p=''):
 if isinstance(v,dict):return {k:x for n,i in v.items() for k,x in leaves(i,f'{p}.{n}' if p else n).items()}
 return {p:v}
old=leaves(OmegaConf.to_container(OmegaConf.load(smoke/'runtime/resolved.yaml'),resolve=True))
new=leaves(OmegaConf.to_container(cfg,resolve=True))
delta={k:{'smoke':old.get(k),'formal':new.get(k)} for k in sorted(old.keys()|new.keys()) if old.get(k)!=new.get(k)}
allowed={'runner.max_epochs','runner.val_check_interval','runner.save_interval','actor.optim.total_training_steps','runner.logger.experiment_name','runner.logger.log_path','algorithm.online_bc.data_path','env.train.task_config.save_path','env.eval.task_config.save_path','env.train.video_cfg.video_base_dir','env.eval.video_cfg.video_base_dir'}
assert set(delta)==allowed,delta
assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
assert cfg.actor.global_batch_size==1024 and cfg.actor.micro_batch_size==32 and cfg.algorithm.update_epoch==10
assert cfg.cluster.component_placement['actor,env,rollout']=='6'
assert cfg.env.train.total_num_envs==32 and cfg.env.train.rollout_epoch==1
assert cfg.env.eval.total_num_envs==8 and cfg.env.eval.rollout_epoch==4
assert cfg.runner.save_interval%cfg.runner.val_check_interval==0
mem={s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemAvailable:','SwapFree:'))}
fs=os.statvfs('/data'); available=fs.f_bavail*fs.f_frsize
assert available>500*2**30,available
out={'time':datetime.datetime.now().astimezone().isoformat(),'head':head,'git_status':status,'run':str(run),'overrides':overrides,'delta':delta,'resolved_yaml':OmegaConf.to_yaml(cfg,resolve=True),'gpu':cmd('nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'),'memory':mem,'disk':cmd('df','-B1','/data','/home'),'data_available_gib':available/2**30,'passed':True}
print(json.dumps(out))
PY
