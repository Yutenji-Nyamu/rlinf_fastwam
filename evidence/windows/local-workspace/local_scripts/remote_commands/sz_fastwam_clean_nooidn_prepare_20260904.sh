#!/usr/bin/env bash
set -euo pipefail
BASE=/data/chenyiteng/projects/rlinf-shenzhen
WT=$BASE/worktrees/fastwam-current-grpo
RT=$BASE/worktrees/robotwin-clean-oidn-off-20260904
BRANCH=codex/sz-robotwin-clean-oidn-off
ROOT=/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo
SLUG=fastwam-grpo-control-fresh100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1
RUN=$ROOT/runs/$SLUG
PACKET=$ROOT/packets/$SLUG
DONOR=$ROOT/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2
VENV=/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
test "$(id -un)" = chenyiteng
test "$(git -C "$WT" rev-parse HEAD)" = 4faade1d50bf21d1caf1b8a4e5f89282a810208a
test -z "$(git -C "$WT" status --porcelain)"
test -z "$(git -C "$WT" diff 7b2331c55d14397cfb4cb16181470ddc8afae44a HEAD -- rlinf examples)"
test "$(cat "$DONOR/runtime/exit_code.txt")" = 255
test ! -e "$RT"
test ! -e "$RUN"
test ! -e "$PACKET"
test "$(git -C "$BASE/RoboTwin-RLinf-support" show --format= --name-only b76d4edcb7e23f09f9de6dafc16a3173f6d07adb)" = envs/_base_task.py
git -C "$BASE/RoboTwin-RLinf-support" worktree add -b "$BRANCH" "$RT" 0008ae6800df9f75fc8de7098bacb01735fd8fd2
git -C "$RT" -c user.name=Yutenji-Nyamu -c user.email=1842710211@qq.com cherry-pick b76d4edcb7e23f09f9de6dafc16a3173f6d07adb
test "$(git -C "$RT" diff --name-only 0008ae6800df9f75fc8de7098bacb01735fd8fd2 HEAD)" = envs/_base_task.py
test "$(git -C "$RT" rev-parse HEAD:robotwin/envs/vector_env.py)" = "$(git -C "$RT" rev-parse 0008ae6800df9f75fc8de7098bacb01735fd8fd2:robotwin/envs/vector_env.py)"
for name in background_texture embodiments objects; do
  test -d "$BASE/RoboTwin-RLinf-support/assets/$name"
  test ! -e "$RT/assets/$name"
  ln -s "$BASE/RoboTwin-RLinf-support/assets/$name" "$RT/assets/$name"
done
git -C "$RT" diff --check 0008ae6800df9f75fc8de7098bacb01735fd8fd2 HEAD
test -z "$(git -C "$RT" status --porcelain)"
mkdir -p "$PACKET" "$RUN/runtime"
export BASE WT RT RUN PACKET DONOR
source "$VENV/bin/activate"
unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export RAY_ADDRESS=172.17.0.1:6389
export ROBOTWIN_PATH="$RT" ROBOT_PLATFORM=ALOHA REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment" RLINF_CODE_WORKING_DIR="$WT"
export PYTHONPATH="$WT:/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/src:$RT"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth
export MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope
export TMPDIR=/home/chenyiteng/cache/fastwam-7faa/tmp
export HF_HOME=/data/chenyiteng/cache/huggingface
export XDG_CACHE_HOME=/data/chenyiteng/cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$WT"
"$VENV/bin/python" - <<'PY'
import os,sys,shlex,subprocess,json,hashlib,ast,shutil
from pathlib import Path
import yaml
packet,run,donor,rt,wt=(Path(os.environ[k]) for k in ['PACKET','RUN','DONOR','RT','WT'])
old=yaml.safe_load((donor/'runtime/resolved.yaml').read_text())
command=shlex.split((donor/'runtime/command.txt').read_text())
exp=run.name.replace('-','_')
resume=0
for i,part in enumerate(command):
 if part.startswith('runner.resume_dir='):
  command[i]='runner.resume_dir=null'; resume+=1
 else:
  command[i]=part.replace(str(donor),str(run)).replace(old['runner']['logger']['experiment_name'],exp)
assert resume==1
command+=['+env.train.task_config.ray_tracing_denoiser=none','+env.eval.task_config.ray_tracing_denoiser=none']
(packet/'command.txt').write_text(shlex.join(command)+'\n')
resolved=subprocess.run(command+['--cfg','job','--resolve'],capture_output=True,text=True)
if resolved.returncode:
 print(resolved.stdout); print(resolved.stderr); raise SystemExit(resolved.returncode)
(packet/'resolved.yaml').write_text(resolved.stdout)
new=yaml.safe_load(resolved.stdout)
def flat(v,p=''):
 if isinstance(v,dict):return {k2:v2 for k,c in v.items() for k2,v2 in flat(c,p+'.'+str(k) if p else str(k)).items()}
 if isinstance(v,list):return {k2:v2 for i,c in enumerate(v) for k2,v2 in flat(c,p+'.'+str(i)).items()}
 return {p:v}
a,b=flat(old),flat(new)
diff={k:[a.get(k),b.get(k)] for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
allowed={'runner.resume_dir','runner.logger.log_path','runner.logger.experiment_name','env.train.task_config.save_path','env.train.video_cfg.video_base_dir','env.eval.task_config.save_path','env.eval.video_cfg.video_base_dir','env.train.task_config.ray_tracing_denoiser','env.eval.task_config.ray_tracing_denoiser'}
assert set(diff)==allowed,(set(diff)-allowed,allowed-set(diff))
assert new['runner']['resume_dir'] is None and new['runner']['max_steps']==100
assert new['env']['train']['total_num_envs']==32 and new['env']['train']['rollout_epoch']==4
assert new['actor']['global_batch_size']==1024 and new['actor']['micro_batch_size']==2
assert new['algorithm']['group_size']==8 and new['algorithm']['update_epoch']==2
assert new['actor']['optim']['lr']==5e-6 and new['actor']['model']['rl']['noise_level']==0.3
assert all(new['env'][m]['task_config']['ray_tracing_denoiser']=='none' for m in ('train','eval'))
for k in ('checkpoint_path','dataset_stats_path'):assert Path(new['actor']['model'][k]).is_file()
ast.parse((rt/'envs/_base_task.py').read_text())
from robotwin.envs import vector_env
import envs._base_task as task
assert Path(vector_env.__file__).resolve()==rt/'robotwin/envs/vector_env.py',vector_env.__file__
assert Path(task.__file__).resolve()==rt/'envs/_base_task.py',task.__file__
heads={name:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip() for name,path in [('rlinf',wt),('robotwin',rt),('fastwam',Path('/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711'))]}
assert heads['fastwam']=='7faa71108368fbb3b6885649f112af607427a2d4'
budget={'outer_steps':100,'train_episodes':12800,'train_submitted_action_slots_max':2457600,'new_query_records_max':102400,'query_record_presentations_max':204800,'actor_optimizer_calls':200,'critic_updates':0,'eval_episodes':640,'checkpoint_generations':10,'wall_timeout_seconds':432000,'reserved_gpu_hours_cap':240,'expected_wall_hours_rough':25,'expected_gpu_hours_rough':50}
contract={'fresh_from_original_sft':True,'donor':str(donor),'resolved_difference_count':len(diff),'allowed_resolved_differences':diff,'source_heads':heads,'robotwin_clean_base':'0008ae6800df9f75fc8de7098bacb01735fd8fd2','old_vector_patch_in_new_worktree':False,'budget':budget,'rendering':'RT/SPP32/pathdepth8 unchanged; explicitly none in train and eval','stop':'step100, driver fatal/exception, or inherited120h timeout; do not stop shared Ray or other jobs'}
(packet/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
(packet/'source_head.txt').write_text('\n'.join(f'{k}={v}' for k,v in heads.items())+'\n')
(packet/'robotwin.patch').write_bytes(subprocess.check_output(['git','-C',str(rt),'diff','0008ae6800df9f75fc8de7098bacb01735fd8fd2','HEAD']))
keys=['PATH','VIRTUAL_ENV','RAY_ADDRESS','ROBOTWIN_PATH','ROBOT_PLATFORM','REPO_PATH','EMBODIED_PATH','RLINF_CODE_WORKING_DIR','PYTHONPATH','DIFFSYNTH_DOWNLOAD_SOURCE','DIFFSYNTH_MODEL_BASE_PATH','MODELSCOPE_CACHE','TMPDIR','HF_HOME','XDG_CACHE_HOME','MUJOCO_GL','PYOPENGL_PLATFORM','HYDRA_FULL_ERROR','PYTHONUNBUFFERED','PYTHONDONTWRITEBYTECODE']
(packet/'environment.sh').write_text('unset CUDA_VISIBLE_DEVICES http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY\n'+''.join(f'export {k}={shlex.quote(os.environ[k])}\n' for k in keys))
(packet/'packet_complete.txt').write_text('source/config/import preflight passed; not launched\n')
files=['command.txt','resolved.yaml','contract.json','source_head.txt','robotwin.patch','environment.sh','packet_complete.txt']
(packet/'sha256.txt').write_text(''.join(f'{hashlib.sha256((packet/n).read_bytes()).hexdigest()}  {n}\n' for n in files))
for n in files+['sha256.txt']:shutil.copyfile(packet/n,run/'runtime'/n)
print('IMPORT_PATHS',vector_env.__file__,task.__file__)
print(json.dumps(contract,ensure_ascii=False,indent=2))
print('EXACT_COMMAND',shlex.join(command))
print('OUTPUT',run)
PY
git -C "$RT" status --short --branch
date -Is
