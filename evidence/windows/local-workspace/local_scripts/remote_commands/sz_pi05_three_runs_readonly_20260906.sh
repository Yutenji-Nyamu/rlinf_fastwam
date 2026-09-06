#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' GIT_OPTIONAL_LOCKS=0
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python -B - <<'PY'
import csv, datetime, json, math, os, re, subprocess, time, xml.etree.ElementTree as ET
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE=Path('/data/chenyiteng/results/rlinf-shenzhen')
TREE=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees')
TZ=datetime.timezone(datetime.timedelta(hours=8))
def now(): return datetime.datetime.now(TZ).isoformat()
def cmd(args, timeout=25):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
        return {'rc':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
    except Exception as e:return {'error':str(e)}
def read(p):
    try:return p.read_text(errors='replace').strip()
    except OSError:return None
def process(pid):
    p=Path('/proc')/str(pid); out={'pid':str(pid),'alive':p.exists()}
    if not out['alive']:return out
    try:
        out['status']={l.split(':',1)[0]:l.split(':',1)[1].strip() for l in (p/'status').read_text().splitlines() if l.startswith(('Name:','State:','Uid:','VmRSS:','VmSize:','Threads:'))}
        out['cmdline']=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        out['open_fds']=len(list((p/'fd').iterdir()))
        out['nofile']=[l for l in (p/'limits').read_text().splitlines() if 'Max open files' in l]
    except OSError as e:out['read_error']=str(e)
    return out
def resource(path):
    rows=list(csv.DictReader(path.open())) if path.is_file() else []
    maxima={}; minima={}
    if rows:
        for key in rows[0]:
            vals=[]
            for row in rows:
                try:
                    v=float(row.get(key,''))
                    if math.isfinite(v):vals.append(v)
                except (ValueError,TypeError):pass
            if vals:maxima[key]=max(vals);minima[key]=min(vals)
    stride=max(1,math.ceil(len(rows)/400))
    sample=rows[::stride]
    if rows and (not sample or sample[-1]!=rows[-1]):sample.append(rows[-1])
    return {'path':str(path),'count':len(rows),'max':maxima,'min':minima,'last':rows[-1] if rows else None,'sample':sample}

specs=[('bc','online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1','pi05-online-bc','',32),
('dvac','online-bc/pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1','pi05-online-bc-dvac','',32),
('grpo','pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1','sidney-pi05-current-rlinf','runtime-resume100-to200',256)]
out={'time':now(),'runs':{},'server':{}}
for name,rel,tree,rtname,budget in specs:
    run=BASE/rel;rt=run/rtname if rtname else run;logpath=rt/'driver.log'
    log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',read(logpath) or '');lines=log.splitlines()
    r={'run':str(run),'runtime':str(rt),'attempts_per_round':budget,'read_at':now(),
       'state':{n:read(rt/n) for n in ['started_at.txt','finished_at.txt','exit_code.txt','wrapper.pid','observer.pid','timeout.pid']},
       'git':{'tree':str(TREE/tree),'head':cmd(['git','-C',str(TREE/tree),'rev-parse','HEAD']),
              'branch':cmd(['git','-C',str(TREE/tree),'branch','--show-current']),
              'status':cmd(['git','-C',str(TREE/tree),'status','--porcelain'])},
       'runtime_source':{str(p):read(p) for p in [run/'runtime/source-head.txt',rt/'source-head.txt'] if p.is_file()},
       'log_mtime':datetime.datetime.fromtimestamp(logpath.stat().st_mtime,TZ).isoformat() if logpath.is_file() else None,
       'step_mentions':re.findall(r'Global Step:\s*(\d+)\s*/\s*(\d+)',log)[-5:],
       'tail':lines[-42:], 'last_table':log[log.rfind('Global Step:'):][-16000:] if 'Global Step:' in log else None}
    r['wrapper']=process(r['state']['wrapper.pid']) if r['state']['wrapper.pid'] else None
    pattern=r'Fatal Python|Traceback \(most recent|OutOfMemoryError|CUDA out of memory|ErrorInitializationFailed|cannot create buffer|pthread_key_create|invalid handle|AssertionError|RuntimeError:|RayActorError|ActorDiedError|Bus error|Segmentation fault'
    hits=[i for i,l in enumerate(lines) if re.search(pattern,l)]
    r['error_hit_count']=len(hits);r['error_contexts']=[{'line':i+1,'text':'\n'.join(lines[max(0,i-5):i+18])} for i in sorted(set(hits[:4]+hits[-3:]))]
    workerids=re.findall(r'((?:Actor|Env|Rollout)Group)\(rank=(\d+)\) pid=(\d+)',log)
    r['workers']={f'{role}-{rank}':process(pid) for role,rank,pid in workerids}
    tbpaths=sorted(set([run/'tensorboard']+[p.parent for p in run.glob('*/*/events.out.tfevents.*')]+[p.parent for p in run.glob('*/events.out.tfevents.*')]))
    scalars={};r['tensorboard_paths']=[];r['tensorboard_errors']=[]
    for tb in tbpaths:
        if not tb.is_dir():continue
        try:
            ea=EventAccumulator(str(tb),size_guidance={'scalars':0});ea.Reload();r['tensorboard_paths'].append(str(tb))
            for tag in ea.Tags()['scalars']:
                scalars.setdefault(tag,[]).extend({'raw_step':p.step,'step':p.step+1,'value':p.value,'wall_time':p.wall_time} for p in ea.Scalars(tag))
        except Exception as e:r['tensorboard_errors'].append(str(e))
    r['scalars']={tag:sorted({p['step']:p for p in sorted(points,key=lambda p:p['wall_time'])}.values(),key=lambda p:p['step']) for tag,points in scalars.items()}
    r['checkpoints']=[]
    ckpts=sorted(run.glob('*/checkpoints/global_step_*'),key=lambda p:int(p.name.rsplit('_',1)[-1]))
    for ck in ckpts:
        files=[{'relative':str(p.relative_to(ck)),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime} for p in ck.rglob('*') if p.is_file()]
        r['checkpoints'].append({'step':int(ck.name.rsplit('_',1)[-1]),'path':str(ck),'bytes':sum(p['bytes'] for p in files),'files':files})
    r['resource']=[resource(p) for p in [run/'resource.csv',rt/'resource.csv'] if p.is_file() and (p!=run/'resource.csv' or not rtname)]
    success=list((run/'success_data').glob('rank_*/*.pt'))
    r['success_data']={'file_count':len(success),'bytes':sum(p.stat().st_size for p in success),'last_names':sorted(str(p.relative_to(run)) for p in success)[-4:]}
    r['dvac_log_tail']=[l for l in lines if 'Online BC DVAC:' in l][-4:]
    out['runs'][name]=r

s=out['server']
s['gpu']=cmd(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit','--format=csv,noheader,nounits'])
s['compute_processes']=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
s['process_owners']={}
for l in s['compute_processes'].get('stdout','').splitlines():
    parts=l.split(',')
    if len(parts)>1 and parts[1].strip().isdigit():
        pid=parts[1].strip();s['process_owners'][pid]=cmd(['ps','-p',pid,'-o','user,pid,ppid,stat,etime,pcpu,pmem,rss,comm'])
s['memory_kib']={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
s['loadavg']=read(Path('/proc/loadavg'));s['cpu_count']=os.cpu_count()
s['pressure']={key:read(Path('/proc/pressure')/key) for key in ['cpu','memory','io']}
s['vmstat']=cmd(['vmstat','1','2'])
s['disks']=cmd(['df','-B1','/','/data','/home']);s['inodes']=cmd(['df','-i','/','/data','/home'])
s['top_cpu']=cmd(['ps','-eo','user,pid,ppid,stat,pcpu,pmem,rss,comm','--sort=-pcpu'])['stdout'].splitlines()[:16]
s['top_rss']=cmd(['ps','-eo','user,pid,ppid,stat,pcpu,pmem,rss,comm','--sort=-rss'])['stdout'].splitlines()[:16]
s['ray_services']=cmd(['ps','-p','321933,322685','-o','user,pid,ppid,stat,etime,comm'])
s['gpu_health']=[]
query=cmd(['nvidia-smi','-q','-x'])
try:
    xml=ET.fromstring(query['stdout'])
    for g in xml.findall('gpu'):
        s['gpu_health'].append({'uuid':g.findtext('uuid'),'ecc_errors':ET.tostring(g.find('ecc_errors'),encoding='unicode') if g.find('ecc_errors') is not None else None,'retired_pages':ET.tostring(g.find('retired_pages'),encoding='unicode') if g.find('retired_pages') is not None else None,'remapped_rows':ET.tostring(g.find('remapped_rows'),encoding='unicode') if g.find('remapped_rows') is not None else None})
except Exception as e:s['gpu_health_error']=str(e)
out['finished_read_at']=now()
print(json.dumps(out,ensure_ascii=False))
PY
