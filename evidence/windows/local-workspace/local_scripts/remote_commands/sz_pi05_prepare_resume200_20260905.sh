#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import copy,datetime,hashlib,json,os,shlex,shutil,subprocess
from pathlib import Path
import yaml
run=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
wt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf')
venv=Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin')
robotwin=Path('/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support')
head='f50e235c5ab1f4390f0ba92bfb13390ed0a86810'
assert subprocess.check_output(['git','-C',str(wt),'rev-parse','HEAD'],text=True).strip()==head
assert not subprocess.check_output(['git','-C',str(wt),'status','--porcelain'],text=True).strip()
rt=run/'runtime-resume100-to200'
assert not rt.exists(), 'Resume packet already exists; inspect rather than overwrite'
base=yaml.safe_load((run/'runtime/resolved.yaml').read_text())
checkpoint=run/base['runner']['logger']['experiment_name']/'checkpoints/global_step_100'
argv=shlex.split((run/'runtime/command.txt').read_text())
assert argv.count('runner.max_steps=100')==1 and argv.count('runner.resume_dir=null')==1
argv=[('runner.max_steps=200' if a=='runner.max_steps=100' else 'runner.resume_dir='+str(checkpoint) if a=='runner.resume_dir=null' else a) for a in argv]
env=os.environ.copy()
for key in ['CUDA_VISIBLE_DEVICES','http_proxy','HTTP_PROXY','https_proxy','HTTPS_PROXY','all_proxy','ALL_PROXY']:
    env.pop(key,None)
fixed={'PATH':str(venv/'bin')+':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin','VIRTUAL_ENV':str(venv),'RAY_ADDRESS':'172.17.0.1:6389','ROBOTWIN_PATH':str(robotwin),'ROBOT_PLATFORM':'ALOHA','REPO_PATH':str(wt),'EMBODIED_PATH':str(wt/'examples/embodiment'),'RLINF_CODE_WORKING_DIR':str(wt),'PYTHONPATH':f'{wt}:{robotwin}:{wt}:{robotwin}','OPENPI_DATA_HOME':'/home/chenyiteng/.cache/openpi','MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','HYDRA_FULL_ERROR':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
env.update(fixed)
result=subprocess.run(argv+['--cfg','job','--resolve'],env=env,cwd=str(wt),capture_output=True,text=True,timeout=60)
assert result.returncode==0, result.stderr[-6000:]
new=yaml.safe_load(result.stdout)
expected=copy.deepcopy(base)
expected['runner']['max_steps']=200
expected['runner']['resume_dir']=str(checkpoint)
assert new==expected, 'Unexpected resolved configuration difference'
rt.mkdir(mode=0o700)
(rt/'resolved.yaml').write_text(result.stdout)
(rt/'compose.stderr.txt').write_text(result.stderr)
(rt/'command.txt').write_text(shlex.join(argv)+'\n')
(rt/'argv.json').write_text(json.dumps(argv,indent=2)+'\n')
(rt/'environment.json').write_text(json.dumps(fixed,indent=2)+'\n')
for name in ['wrapper.sh','observer.sh']:
    shutil.copyfile(run/'runtime'/name,rt/name)
    (rt/name).chmod(0o700)
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'runtime/command.txt',run/'runtime/resolved.yaml',wt/'rlinf/utils/metric_logger.py',wt/'rlinf/envs/wrappers/record_video.py',wt/'rlinf/runners/embodied_runner.py']}
contract={'prepared_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'run':str(run),'runtime':str(rt),'worktree':str(wt),'source_head':head,'resume_dir':str(checkpoint),'config_diff':{'runner.max_steps':[100,200],'runner.resume_dir':[None,str(checkpoint)]},'all_other_resolved_leaves_equal':True,'physical_gpus':[4,5],'new_steps':100,'new_trajectories':25600,'new_optimizer_calls':200,'new_eval_episodes':640,'new_checkpoints':10,'max_new_query_records':102400,'max_new_action_slots':5120000,'estimated_hours':[41,42],'estimated_gpu_hours':[82,84],'estimated_checkpoint_bytes':288282105700,'baseline_hashes':hashes}
(rt/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
(rt/'launch.py').write_text(r'''import datetime,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
rt=Path(__file__).resolve().parent
c=json.loads((rt/'contract.json').read_text())
run=Path(c['run']); wt=Path(c['worktree']); old=run/'runtime'
if (rt/'launch_attempt.json').exists():
    print('ALREADY_ATTEMPTED: inspect resume driver and wrapper; never duplicate launch')
    sys.exit(0)
finished=old/'finished_at.txt'; exitfile=old/'exit_code.txt'
if not finished.exists() or not exitfile.exists():
    print('WAITING: original formal100 is not finished')
    sys.exit(0)
assert exitfile.read_text().strip()=='0', 'Original run failed; no automatic fallback'
oldpid=old.joinpath('wrapper.pid').read_text().strip()
if Path('/proc',oldpid).exists():
    print('WAITING: original wrapper has not exited')
    sys.exit(0)
assert subprocess.check_output(['git','-C',str(wt),'rev-parse','HEAD'],text=True).strip()==c['source_head']
assert not subprocess.check_output(['git','-C',str(wt),'status','--porcelain'],text=True).strip()
for p,h in c['baseline_hashes'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, ('Baseline changed',p)
checkpoint=Path(c['resume_dir'])
rel=['actor/local_shard_checkpoint/checkpoint_rank_0.pt','actor/local_shard_checkpoint/checkpoint_rank_1.pt','actor/model_state_dict/full_weights.pt']
files=[]
for name in rel:
    p=checkpoint/name
    assert p.is_file() and p.stat().st_size>0, ('Missing checkpoint',str(p))
    with zipfile.ZipFile(p) as z:
        assert any(n.endswith('/data.pkl') for n in z.namelist()), str(p)
    files.append({'path':str(p),'bytes':p.stat().st_size})
assert shutil.disk_usage(run).free>c['estimated_checkpoint_bytes'], 'Insufficient space for authorized checkpoint budget'
gpu=subprocess.run(['nvidia-smi','-i','4,5','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
if gpu.stdout.strip():
    print('WAITING: GPU4/5 still has compute processes; no process intervention')
    sys.exit(0)
env=os.environ.copy()
for key in ['CUDA_VISIBLE_DEVICES','http_proxy','HTTP_PROXY','https_proxy','HTTPS_PROXY','all_proxy','ALL_PROXY']:
    env.pop(key,None)
env.update(json.loads((rt/'environment.json').read_text()))
subprocess.run([str(Path(env['VIRTUAL_ENV'])/'bin/ray'),'status'],env=env,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=30)
video=run/'video/eval'; archive=run/'video/eval_steps001_100'
assert not archive.exists(), 'Video archive already exists; inspect before retry'
assert video.is_dir() and video.resolve().is_relative_to(run.resolve())
assert archive.parent.resolve().is_relative_to(run.resolve())
assert not any((run/'robotwin_data').rglob('*')), 'Unexpected original raw data; preserve before resume'
attempt={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'original_finished':finished.read_text().strip(),'original_exit':0,'checkpoint_files':files,'gpu45_compute_before_launch':gpu.stdout.strip(),'free_bytes':shutil.disk_usage(run).free}
with (rt/'launch_attempt.json').open('x') as f:json.dump(attempt,f,indent=2)
shutil.copy2(run/'tensorboard/config.yaml',rt/'tensorboard_config_steps001_100.yaml')
video.rename(archive)
with (rt/'wrapper.log').open('xb') as out:
    p=subprocess.Popen(['bash',str(rt/'wrapper.sh'),str(rt),str(rt/'command.txt')],env=env,cwd=str(wt),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
(rt/'wrapper.pid').write_text(str(p.pid)+'\n');(rt/'owned.pgid').write_text(str(p.pid)+'\n')
with (rt/'observer.log').open('xb') as out:
    obs=subprocess.Popen(['bash',str(rt/'observer.sh'),str(p.pid),str(rt/'resource.csv')],env=env,cwd=str(wt),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
(rt/'observer.pid').write_text(str(obs.pid)+'\n')
print(json.dumps({'status':'LAUNCHED','runtime':str(rt),'wrapper_pid':p.pid,'observer_pid':obs.pid,'resume_dir':str(checkpoint),'max_steps':200,'only_two_config_changes':True}))
''')
print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','contract':contract,'command':(rt/'command.txt').read_text(),'resolved_yaml':result.stdout}))
PY
