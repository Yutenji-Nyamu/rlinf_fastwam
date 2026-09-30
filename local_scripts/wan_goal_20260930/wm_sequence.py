"""OFT smoke -> audited PI05 head-only smoke -> formal PI05, with isolated Ray per stage."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path('/data/chenyiteng/projects/wan-goal-sz3')
sys.path.insert(0,str(ROOT/'scripts/resource_switch'))
from common import account,atomic
from wm_stage import run_stage
from cleanup_owned import gpu_processes

account()
run=Path(os.environ['WAN_GOAL_RUN_DIR'])
cycle=ROOT/'cycles'/os.environ['WAN_GOAL_CYCLE_ID']
cycle.mkdir(parents=True,exist_ok=True)
events=run/'sequence-events.jsonl'
stop=False
import signal
def stop_sequence(signum,frame):
    global stop
    stop=True
signal.signal(signal.SIGTERM,stop_sequence)
signal.signal(signal.SIGINT,stop_sequence)

def state(phase,**details):
    record={'time':time.time(),'phase':phase,**details}
    atomic(run/'sequence-current.json',record)
    with events.open('a') as out:out.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)

base_env={
 'TMPDIR':str(ROOT/'tmp'), 'HF_HOME':str(ROOT/'cache/huggingface'),
 'XDG_CACHE_HOME':str(ROOT/'cache/xdg'), 'TORCH_HOME':str(ROOT/'cache/torch'),
 'TRITON_CACHE_DIR':str(ROOT/'cache/triton'), 'TORCHINDUCTOR_CACHE_DIR':str(ROOT/'cache/torchinductor'),
 'OPENPI_DATA_HOME':str(ROOT/'models/openpi-assets'), 'ROBOT_PLATFORM':'LIBERO',
 'LIBERO_TYPE':'standard', 'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl',
 'WAN_GOAL_WM_PATH':str(ROOT/'models/wan-goal'),
 'WAN_GOAL_OFT_PATH':str(ROOT/'models/oft-goal'),
 'WAN_GOAL_PI05_PATH':str(ROOT/'models/pi05-libero'),
 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','WAN_PATH':str(ROOT/'src/diffsynth-studio'),
 'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false',
}
stages=[
 ('oft-smoke','oft-wan','RLinf','wan_goal_oft_smoke_sz3',63843,'o1',True),
 ('pi05-smoke','pi05-wan','RLinf-pi05','wan_goal_pi05_headonly_smoke_sz3',63844,'p1',True),
 ('pi05-formal','pi05-wan','RLinf-pi05','wan_goal_pi05_headonly_formal_sz3',63845,'p2',False),
]
for name,env_name,repo_name,config,port,suffix,verify in stages:
    if stop:raise RuntimeError('Sequence cancelled before next stage')
    stage_run=run/name
    attempt=run/(name+'-control');attempt.mkdir()
    py=ROOT/'envs'/env_name/'bin/python'
    namespace='wan_goal_'+run.name.replace('-','_')+'_'+name.replace('-','_')
    # Short unique Unix socket path. The outer run name itself is longer.
    short_id=hashlib.sha256(str(run).encode()).hexdigest()[:8]
    ray_tmp=f'/data/chenyiteng/wr/{short_id}{suffix}'
    env={**base_env,'LIBERO_CONFIG_PATH':str(ROOT/'config'/('libero-pi05' if 'pi05' in name else 'libero'))}
    spec={
      'schema':1,'run_dir':str(stage_run),'cwd':str(ROOT/repo_name),
      'physical_gpus':[4,5,6,7],'max_seconds':None,'cleanup_timeout_seconds':180,
      'command':[str(py),'-u',str(ROOT/'scripts/private_ray_driver.py'),
                 '--repo',str(ROOT/repo_name),'--config',config,'--log-dir',str(stage_run),'--port',str(port)],
      'cleanup_command':[sys.executable,'-u',str(ROOT/'scripts/resource_switch/cleanup_owned.py')],
      'environment':env,
      'ray':{'isolation':'dedicated','address':f'127.0.0.1:{port}','namespace':namespace,'temp_dir':ray_tmp},
    }
    atomic(attempt/'spec.json',spec)
    state('STARTING_'+name.upper().replace('-','_'),config=config,stage_run=str(stage_run))
    released=run_stage(spec,cycle,attempt,lambda _:gpu_processes(),state,lambda:stop)
    if released['outcome']!='completed' or released['wm_exit_code']!=0:
        raise RuntimeError(f'{name} failed; GPUs released; returning to Dojo. See {stage_run}')
    if verify:
        verify_env=os.environ.copy();verify_env.update(base_env,CUDA_VISIBLE_DEVICES='')
        argv=[str(py),str(ROOT/'scripts/verify_smoke.py'),'--run-dir',str(stage_run),
              '--experiment',config,'--output',str(attempt/'smoke-verification.json')]
        state('VERIFYING_'+name.upper().replace('-','_'))
        with (attempt/'verification.log').open('w') as out:
            subprocess.run(argv,env=verify_env,stdout=out,stderr=subprocess.STDOUT,check=True)
        state('VERIFIED_'+name.upper().replace('-','_'))
state('FORMAL_COMPLETE_RETURNING_TO_DOJO')
