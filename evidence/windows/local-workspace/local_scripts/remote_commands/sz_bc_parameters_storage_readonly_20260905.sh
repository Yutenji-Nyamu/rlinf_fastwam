set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
nice -n 19 ionice -c 3 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import csv,datetime,hashlib,json,os,re,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
tz=datetime.timezone(datetime.timedelta(hours=8))
base=Path('/data/chenyiteng'); trees=base/'projects/rlinf-shenzhen/worktrees'
run=base/'results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1'
rt=run/'runtime-resume100-to200'
def cmd(args,timeout=30):
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    return {'rc':p.returncode,'out':p.stdout.strip(),'err':p.stderr.strip()[:1600]}
def read(p):
    try:return Path(p).read_text(errors='replace')
    except FileNotFoundError:return None
def storage():
    d=cmd(['du','-x','-B1','--max-depth=5','--',str(base)],600)
    rows=[]
    for line in d['out'].splitlines():
        size,path=line.split('\t',1)
        if int(size)>=100*1024**2 or Path(path).parent==base:
            rows.append({'path':path,'allocated_bytes':int(size)})
    checkpoints=[]
    for current,dirs,files in os.walk(base/'results',followlinks=False):
        dirs[:]=[x for x in dirs if x not in ('video','videos','.git','success_data','tensorboard')]
        if 'checkpoints' not in dirs:continue
        cr=Path(current)/'checkpoints';gens=[]
        for p in cr.iterdir():
            if not p.is_dir() or not re.fullmatch(r'global_step_\d+',p.name):continue
            fs=[]
            for f in p.rglob('*'):
                if not f.is_file() or f.is_symlink():continue
                st=f.stat()
                fs.append({'file':str(f.relative_to(p)),'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'nlink':st.st_nlink})
            gens.append({'step':int(p.name.split('_')[-1]),'path':str(p),'files':fs})
        checkpoints.append({'root':str(cr),'generations':sorted(gens,key=lambda p:p['step'])})
        dirs.remove('checkpoints')
    return {'du_rc':d['rc'],'du_errors':d['err'],'du':rows,'checkpoint_runs':checkpoints,'finished':datetime.datetime.now(tz).isoformat()}
result={'time':datetime.datetime.now(tz).isoformat(),'scope':'Read-only own metadata, fixed source and existing metrics; no model forward or mutations.'}
with ThreadPoolExecutor(max_workers=1) as pool:
    sf=pool.submit(storage)
    result['gpu']=cmd(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'])
    result['meminfo']=read('/proc/meminfo');result['pressure']={x:read('/proc/pressure/'+x) for x in ('memory','io','cpu')}
    result['vmstat']=cmd(['vmstat','1','3']);result['disk']=cmd(['df','-B1','/data','/home'])
    state={k:read(rt/k) for k in ('wrapper.pid','started_at.txt','finished_at.txt','exit_code.txt')}
    log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',read(rt/'driver.log') or '')
    keys=('Fatal Python error','CUDA out of memory','OutOfMemoryError','Traceback (most recent call last):','RuntimeError:','pthread_key_create','OIDN Error')
    sid={'run':str(run),'runtime':str(rt),'state':state,'wrapper_alive':bool(state['wrapper.pid']) and Path('/proc',state['wrapper.pid'].strip()).exists(),
         'completed_steps':re.findall(r'Global Step:\s*(\d+)\s*/',log)[-3:],
         'phase':[s for s in log.splitlines() if 'Generating Rollout Epochs:' in s or 'Evaluating Rollout Epochs:' in s][-2:],
         'errors':{k:log.count(k) for k in keys},'scalars':{},'resource':{}}
    ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
    for tag in ea.Tags().get('scalars',[]):
        if any(x in tag for x in ('success_once','loss','grad_norm','approx_kl','clip_fraction','time/step','time/actor_training','time/generate_rollouts','time/eval')):
            sid['scalars'][tag]=[{'step':p.step+1,'value':p.value,'wall_time':p.wall_time} for p in ea.Scalars(tag)]
    for r in (run/'runtime',rt):
        f=r/'resource.csv'
        if f.is_file():
            rows=list(csv.DictReader(f.open())); stride=max(1,len(rows)//400)
            sid['resource'][r.name]={'rows':rows[::stride]+(rows[-1:] if len(rows)%stride else []),'raw_count':len(rows)}
    result['sidney']=sid
    result['git']={n:{'head':cmd(['git','-C',str(trees/n),'rev-parse','HEAD']),
                      'status':cmd(['git','-C',str(trees/n),'status','--porcelain'])} for n in ('pi0-online-bc','pi05-online-bc','sidney-pi05-current-rlinf')}
    selected={
      'pi05_config':trees/'pi05-online-bc/examples/embodiment/config/online_bc_model/pi05_sidney.yaml',
      'bc_base_config':trees/'pi05-online-bc/examples/embodiment/config/robotwin_adjust_bottle_online_bc_openpi.yaml',
      'official_pi0_dagger':trees/'pi05-online-bc/examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml',
      'official_pi05_sft':trees/'pi05-online-bc/examples/sft/config/robotwin_sft_openpi_pi05.yaml',
      'replay':trees/'pi05-online-bc/rlinf/data/online_bc.py',
      'bc_actor':trees/'pi05-online-bc/rlinf/workers/actor/fsdp_online_bc_policy_worker.py',
      'runner':trees/'pi05-online-bc/rlinf/runners/embodied_runner.py',
      'runner_utils':trees/'pi05-online-bc/rlinf/utils/runner_utils.py',
      'bc_smoke_resolved':base/'results/rlinf-shenzhen/online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1/runtime/resolved.yaml',
      'pi0_formal_resolved':base/'results/rlinf-shenzhen/online-bc/pi0-adjust-bottle-bc32x1-b1024-u10-eval16x2-gpu6-formal100-20260905-v1/resolved.yaml'}
    result['source']={}
    for name,p in selected.items():
        content=read(p)
        result['source'][name]={'path':str(p),'content':content,'sha256':hashlib.sha256(content.encode()).hexdigest() if content else None}
    result['retention_search']=cmd(['git','-C',str(trees/'pi05-online-bc'),'grep','-n','-E','save_total_limit|keep_last|keep_checkpoint|rotate_checkpoints|save_full_weights','--','rlinf/runners','rlinf/workers/actor','rlinf/utils'])
    result['storage']=sf.result()
result['end_time']=datetime.datetime.now(tz).isoformat()
print(json.dumps(result,ensure_ascii=False))
PY
