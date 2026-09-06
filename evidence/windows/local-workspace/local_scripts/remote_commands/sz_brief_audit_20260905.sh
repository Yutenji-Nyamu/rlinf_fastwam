#!/usr/bin/env bash
set -eu
PYTHONDONTWRITEBYTECODE=1 /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import datetime, json, os, pwd, re, subprocess
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def cmd(args, timeout=15):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
        return {'rc':p.returncode,'out':p.stdout.strip(),'err':p.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {'rc':'timeout'}
def read(p):
    try: return Path(p).read_text(errors='replace').strip()
    except OSError: return None
def stamp(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

s={'time':stamp(),'identity':cmd(['id']),'hostname':cmd(['hostname']), 'cpu_count':os.cpu_count(), 'os':read('/etc/os-release')}
s['gpu']=cmd(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'])
s['gpu_processes']=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory,process_name','--format=csv,noheader,nounits'])
s['gpu_owners']=[]
for line in s['gpu_processes'].get('out','').splitlines():
    fields=[x.strip() for x in line.split(',')]
    try:
        p=Path('/proc')/fields[1]
        s['gpu_owners'].append({'pid':int(fields[1]),'user':pwd.getpwuid(p.stat().st_uid).pw_name,'comm':read(p/'comm')})
    except (OSError,IndexError,ValueError): pass
s['meminfo']=read('/proc/meminfo'); s['loadavg']=read('/proc/loadavg');s['uptime']=read('/proc/uptime')
s['pressure']={k:read('/proc/pressure/'+k) for k in ['cpu','memory','io']}
s['vmstat']=cmd(['vmstat','1','3'])
s['disk']=cmd(['df','-B1','/','/home','/data']);s['inodes']=cmd(['df','-i','/','/home','/data'])
s['services']=cmd(['systemctl','is-active','ssh','mihomo'])
s['failed_services']=cmd(['systemctl','--failed','--no-pager','--plain'])
s['gpu_health']=cmd(['nvidia-smi','-q','-d','ECC,ROW_REMAPPER'])
s['top_rss']=cmd(['ps','-eo','user:16,pid,ppid,stat,%cpu,rss,comm','--sort=-rss'])
s['top_rss']['out']='\n'.join(s['top_rss']['out'].splitlines()[:14])
s['core_processes']=cmd(['ps','-p','321933,322685,3176215,1568973,262234,262238','-o','user:16,pid,ppid,etime,stat,%cpu,rss,comm'])
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
runs={
 'sidney':base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1',
 'fastwam':base/'fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3',
 'online_bc':base/'online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v2'}
s['runs']={}
errors=['OIDN Error','pthread_key_create failed','Fatal Python error','CUDA out of memory','OutOfMemoryError','Traceback','RayActorError','Segmentation fault','RuntimeError:','Exiting main process due to a failure']
for name,root in runs.items():
    rt=root/'runtime' if (root/'runtime/driver.log').is_file() else root
    d={'path':str(root),'runtime':str(rt),'state':{k:read(rt/k) for k in ['wrapper.pid','driver.pid','timeout.pid','started_at.txt','finished_at.txt','exit_code.txt']}}
    f=rt/'driver.log'
    log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',read(f) or '')
    lines=log.splitlines();steps=re.findall(r'Global Step:\s*(\d+)\s*/',log)
    d['last_global_step']=int(steps[-1]) if steps else None
    d['log_mtime']=datetime.datetime.fromtimestamp(f.stat().st_mtime,datetime.timezone.utc).isoformat() if f.exists() else None
    d['log_tail']=lines[-22:]
    d['errors']={k:log.count(k) for k in errors}
    d['error_lines']=[l[:700] for l in lines if any(k in l for k in errors)][:8]
    d['metrics_lines']=[l[-1500:] for l in lines if any(k in l for k in ['Global Step:', 'success_rate', 'train/loss', 'bc_loss','online_bc/','Saving checkpoint','Saved checkpoint'])][-12:]
    d['scalars']={}
    if (root/'tensorboard').exists():
        try:
            ea=EventAccumulator(str(root/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
            for tag in ea.Tags().get('scalars',[]):
                if any(k in tag for k in ['success','loss','grad_norm','time/step','approx_kl','clip_fraction','pool','episodes','update']):
                    ev=ea.Scalars(tag)
                    d['scalars'][tag]={'count':len(ev),'first':[[p.step,p.value] for p in ev[:3]],'last':[[p.step,p.value] for p in ev[-10:]]}
        except Exception as e:d['scalar_error']=str(e)
    d['checkpoint_latest']=[]
    ckroots=[root/'checkpoints']+[p/'checkpoints' for p in root.iterdir() if p.is_dir() and (p/'checkpoints').is_dir()]
    for cr in ckroots:
        steps=sorted(cr.glob('global_step_*'),key=lambda p:int(p.name.rsplit('_',1)[1]))
        if steps:
            p=steps[-1]
            d['checkpoint_latest'].append({'path':str(p),'files':[{'path':str(f.relative_to(p)),'bytes':f.stat().st_size} for f in p.rglob('*') if f.is_file()]})
    d['root_names']=sorted(p.name for p in root.iterdir())
    s['runs'][name]=d
s['owned_drivers']=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        if p.stat().st_uid!=os.getuid():continue
        raw=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        if raw and ('python' in raw[0] or raw[0].startswith('ray::')) and any('train_embodied' in x or 'train_online_bc' in x for x in raw):
            s['owned_drivers'].append({'pid':int(p.name),'command':' '.join(raw)[:800]})
    except OSError:pass
s['end_time']=stamp()
print(json.dumps(s,ensure_ascii=False,allow_nan=False))
PY
