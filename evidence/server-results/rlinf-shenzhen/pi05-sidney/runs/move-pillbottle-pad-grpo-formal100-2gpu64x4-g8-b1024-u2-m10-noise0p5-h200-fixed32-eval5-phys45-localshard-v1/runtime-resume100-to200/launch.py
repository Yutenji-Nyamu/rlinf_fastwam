import datetime,hashlib,json,os,shutil,subprocess,sys,zipfile
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
