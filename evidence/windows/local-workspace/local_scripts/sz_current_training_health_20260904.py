"""Read-only server snapshot. Stream over SSH stdin; writes no remote artifacts."""
import csv, datetime, io, json, os, re, statistics, subprocess, time
from pathlib import Path
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TZ=datetime.timezone(datetime.timedelta(hours=8))
def stamp(t=None): return datetime.datetime.fromtimestamp(time.time() if t is None else t,TZ).isoformat()
def cmd(args,timeout=15):
    try:
        p=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
        return {'rc':p.returncode,'out':p.stdout,'err':p.stderr}
    except subprocess.TimeoutExpired: return {'rc':'timeout','out':'','err':str(args[:2])}
def txt(path):
    try:return Path(path).read_text(errors='replace').strip()
    except OSError as e:return 'UNAVAILABLE: '+str(e)

snap={'time':stamp(),'identity':cmd(['id']),'hostname':cmd(['hostname']),'runs':{},'health':{},'git':{}}
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
runs={
 'fastwam':Path(os.environ.get('FASTWAM_RUN_PATH', str(base/'fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1'))),
 'sidney':base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1'}
ansi=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
errors=['OIDN Error','pthread_key_create failed','Fatal Python error','CUDA out of memory','OutOfMemoryError','Traceback','RayActorError','Segmentation fault','RuntimeError:','Exception occurred while running','Exiting main process due to a failure']
for name,run in runs.items():
    rt=run/'runtime'; cfg=yaml.safe_load((rt/'resolved.yaml').read_text())
    log=ansi.sub('',(rt/'driver.log').read_text(errors='replace')); lines=log.splitlines()
    steps=[int(x) for x in re.findall(r'Global Step:\s*(\d+)\s*/',log)]
    state={k:txt(rt/k) if (rt/k).exists() else None for k in ['wrapper.pid','observer.pid','started_at.txt','exit_code.txt','finished_at.txt']}
    state['wrapper_alive']=bool(state['wrapper.pid'] and Path('/proc',state['wrapper.pid']).exists())
    state.update(completed_step=steps[-1] if steps else 0,log_mtime=stamp((rt/'driver.log').stat().st_mtime),log_bytes=(rt/'driver.log').stat().st_size)
    state['phase']=[s[-500:] for s in lines if any(k in s for k in ['Generating Rollout Epochs:','Global Step:','Evaluating Epochs:','Saving checkpoint','Saved checkpoint','Train Epoch:'])][-10:]
    state['errors']={k:log.count(k) for k in errors}
    hit=[i for i,l in enumerate(lines) if any(k in l for k in errors)]
    state['error_context']=lines[max(0,hit[0]-12):min(len(lines),hit[0]+35)] if hit else []
    state['tail']=lines[-22:]
    scalars={}; scalar_error=None
    try:
        ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
        tags=ea.Tags().get('scalars',[])
        for tag in tags:
            if any(k in tag for k in ['success','grad_norm','approx_kl','clip_fraction','importance_ratio','time/step','time/rollout','time/train','actor/total_loss','actor/lr']):
                scalars[tag]=[{'step':p.step,'value':p.value,'time':p.wall_time} for p in ea.Scalars(tag)]
    except Exception as e: scalar_error=repr(e); tags=[]
    resource=list(csv.DictReader(io.StringIO(txt(rt/'resource.csv')))) if (rt/'resource.csv').exists() else []
    ckroots=sorted([p for p in run.iterdir() if p.is_dir() and (p/'checkpoints').is_dir()])
    checkpoints=[]
    for cr in ckroots:
        for stepdir in sorted((cr/'checkpoints').glob('global_step_*'),key=lambda p:int(p.name.rsplit('_',1)[1])):
            files=[]
            for p in stepdir.rglob('*'):
                if p.is_file():files.append({'path':str(p.relative_to(stepdir)),'bytes':p.stat().st_size,'mtime':stamp(p.stat().st_mtime)})
            checkpoints.append({'step':int(stepdir.name.rsplit('_',1)[1]),'path':str(stepdir),'files':files})
    # Some older/current runners place checkpoints directly under run.
    if (run/'checkpoints').is_dir():
        for stepdir in sorted((run/'checkpoints').glob('global_step_*')):
            checkpoints.append({'step':int(stepdir.name.rsplit('_',1)[1]),'path':str(stepdir),'files':[{'path':str(p.relative_to(stepdir)),'bytes':p.stat().st_size} for p in stepdir.rglob('*') if p.is_file()]})
    snap['runs'][name]={'path':str(run),'state':state,'config':cfg,'scalar_tags':tags,'scalars':scalars,'scalar_error':scalar_error,'resource':resource,'checkpoints':checkpoints}

h=snap['health']
h['gpu']=cmd(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit','--format=csv,noheader,nounits'])
h['gpu_processes']=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory,process_name','--format=csv,noheader,nounits'])
h['gpu_health']=cmd(['nvidia-smi','-q','-d','ECC,PAGE_RETIREMENT,ROW_REMAPPER,TEMPERATURE'])
h['meminfo']=txt('/proc/meminfo');h['loadavg']=txt('/proc/loadavg')
h['cpu_count']=os.cpu_count();h['uptime']=txt('/proc/uptime')
h['pressure']={k:txt('/proc/pressure/'+k) for k in ['cpu','memory','io']}
h['vmstat']=cmd(['vmstat','1','3'])
h['disk']=cmd(['df','-B1','/','/home','/data']);h['inodes']=cmd(['df','-i','/','/home','/data'])
h['block_devices']=cmd(['lsblk','-d','-o','NAME,MODEL,SIZE,ROTA'])
h['core_processes']=cmd(['ps','-p','321933,322685,3176215,'+os.environ.get('FASTWAM_DRIVER_PID','1052633'),'-o','user,pid,pgid,etime,stat,%cpu,rss,comm'])
h['top_rss']=cmd(['ps','-eo','user,pid,ppid,pgid,stat,%cpu,rss,comm','--sort=-rss']);h['top_rss']['out']='\n'.join(h['top_rss']['out'].splitlines()[:16])
h['services']=cmd(['systemctl','is-active','ssh','mihomo'])
h['failed_services']=cmd(['systemctl','--failed','--no-pager','--plain'])
h['kernel_recent']=cmd(['journalctl','-k','--since','today','--no-pager','-n','300'])
h['listeners']=cmd(['ss','-lnt']);h['listeners']['out']='\n'.join(l for l in h['listeners']['out'].splitlines() if re.search(r':(22|6389|7890)\b',l))
# Only inspect this user's process details; other users are represented by non-sensitive process summaries.
owned=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        if p.stat().st_uid!=os.getuid():continue
        c=(p/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')
        if not (c.startswith('ray::') or ('train_embodied_agent.py' in c and 'bash -c' not in c)):continue
        env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
        rt=env.get('ROBOTWIN_PATH'); gpu=env.get('CUDA_VISIBLE_DEVICES')
        status=txt(p/'status');rss=re.search(r'^VmRSS:\s+(\d+)',status,re.M);threads=re.search(r'^Threads:\s+(\d+)',status,re.M)
        if 'Env' in c or 'Rollout' in c or 'Actor' in c or 'train_embodied' in c:
            owned.append({'pid':int(p.name),'command':c[:150],'robotwin':rt,'gpu':gpu,'rss_kib':int(rss[1]) if rss else None,'threads':int(threads[1]) if threads else None,
                          'namespace':env.get('CLUSTER_NAMESPACE'),'ld_preload':env.get('LD_PRELOAD'),
                          'scene_fence_mapped': 'librlinf_scene_fence.so' in txt(p/'maps')})
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
h['owned_workers']=owned
repo_base=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
for name in ['fastwam-current-grpo','robotwin-clean-oidn-off-20260904','sidney-pi05-current-rlinf']:
    repo=repo_base/name
    snap['git'][name]={'head':cmd(['git','--no-optional-locks','-C',str(repo),'rev-parse','HEAD']),'dirty':cmd(['git','--no-optional-locks','-C',str(repo),'status','--porcelain'])}
snap['end_time']=stamp()
payload=json.dumps(snap,ensure_ascii=False,allow_nan=False)
if os.environ.get('SNAPSHOT_OUTPUT_COMPRESSED') == '1':
    import base64, zlib
    print('SNAPSHOT_ZLIB '+base64.b64encode(zlib.compress(payload.encode(),9)).decode())
else:
    print('SNAPSHOT_JSON '+payload)
