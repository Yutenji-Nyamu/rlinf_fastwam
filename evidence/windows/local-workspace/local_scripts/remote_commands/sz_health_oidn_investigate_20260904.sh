#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import subprocess,os,json,re,datetime,collections,hashlib
def run(label,args,timeout=20):
 print('\nSECTION',label,flush=True)
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
  print(r.stdout); print('rc=',r.returncode,'stderr=',r.stderr[-2000:])
 except Exception as e: print(type(e).__name__,str(e))
print('AUDIT_CST',datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
run('identity',['id']);run('uptime',['uptime']);run('CPU',['lscpu'])
run('vmstat_3_samples',['vmstat','1','3']);run('memory',['free','-b'])
run('disk_space',['df','-B1','/','/home','/data','/dev/shm']);run('inodes',['df','-i','/','/home','/data'])
run('mounts',['findmnt','-o','TARGET,SOURCE,FSTYPE,OPTIONS','--target','/data'])
for f in ('/proc/pressure/cpu','/proc/pressure/memory','/proc/pressure/io','/proc/sys/kernel/threads-max','/proc/sys/kernel/pid_max','/proc/sys/fs/file-nr'):
 print(f,Path(f).read_text().strip())
run('keys_max',['getconf','PTHREAD_KEYS_MAX'])
run('GPU',['nvidia-smi','--query-gpu=index,uuid,driver_version,pstate,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'])
run('GPU_processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory,process_name','--format=csv,noheader,nounits'])
run('GPU_errors',['nvidia-smi','-q','-d','ECC,PAGE_RETIREMENT,ROW_REMAPPER'])
run('failed_services',['systemctl','--failed','--no-pager','--no-legend'])
run('kernel_warning_visible',['journalctl','-k','--since','2026-09-03 14:45:00 UTC','--until','2026-09-03 16:10:00 UTC','-p','warning','--no-pager','-n','80'])
run('interfaces',['ip','-s','link'])
run('listening_local',['ss','-lntp'])
ps=subprocess.run(['ps','-e','-o','user=,stat=,nlwp=,rss=,pcpu=,comm='],capture_output=True,text=True).stdout
agg={}; zombies=0
for l in ps.splitlines():
 a=l.split(None,5)
 if len(a)!=6: continue
 u,s,n,r,c,comm=a; v=agg.setdefault(u,{'processes':0,'threads':0,'rss_gib':0,'cpu_sum':0,'zombies':0})
 v['processes']+=1;v['threads']+=int(n);v['rss_gib']+=int(r)/2**20;v['cpu_sum']+=float(c);v['zombies']+=int('Z' in s)
print('PROCESS_USER_AGG_RSS_SUM_NOT_UNIQUE',json.dumps(agg))
run('owned_top_rss',['ps','-u',str(os.getuid()),'--sort=-rss','-o','pid,ppid,pgid,etime,stat,nlwp,%cpu,rss,comm'])
base=Path('/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs')
slugs=['fastwam-grpo-control-formal100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-v1','fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2']
for slug in slugs:
 root=base/slug; p=root/'runtime/driver.log'
 lines=p.read_text(errors='replace').splitlines()
 print('\nDRIVER',slug,'BYTES',p.stat().st_size)
 patterns=['OIDN Error: invalid handle','OIDN Error: pthread_key_create failed','Fatal Python error','Traceback','CUDA out of memory','OutOfMemoryError','Saving checkpoint']
 for pat in patterns:
  hits=[(i+1,l[:700]) for i,l in enumerate(lines) if pat in l]
  print('PATTERN',pat,'COUNT',len(hits),'FIRST',hits[:2],'LAST',hits[-1:])
 first=next((i for i,l in enumerate(lines) if 'OIDN Error:' in l),None)
 if first is not None: print('FIRST_ERROR_CONTEXT','\n'.join(f'{i+1}: {lines[i]}' for i in range(max(0,first-55),min(len(lines),first+5))))
 for name in ('exit_code.txt','finished_at.txt'):
  p=root/'runtime'/name;print(name,p.read_text() if p.exists() else 'absent')
 res=(root/'runtime/resource.csv').read_text().splitlines()
 print('RESOURCES_AROUND_FAILURE','\n'.join([res[0]]+res[-8:]))
ray=Path('/data/chenyiteng/ray/rlt-dsrl-v3/session_2026-08-23_16-27-53_911161_321906/logs')
for pid in (3590591,3590594,2813140,2813142):
 for parent in (ray,ray/'old'):
  for p in parent.glob(f'worker-*-{pid}.err'):
   print('\nRAY_WORKER',str(p),'bytes',p.stat().st_size)
   lines=p.read_text(errors='replace').splitlines()
   print('HEAD','\n'.join(lines[:8]))
   start=next((i for i,l in enumerate(lines) if 'Fatal Python error' in l),None)
   if start is not None: print('FATAL_STACK','\n'.join(f'{i+1}: {lines[i]}' for i in range(start,min(len(lines),start+180))))
   print('PATH_COUNTS',json.dumps(dict(collections.Counter(re.findall(r'File "([^"]+(?:vector_env|_base_task|robotwin_env)\.py)"','\n'.join(lines))))))
rt=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-vector-render-lifecycle-fix-0008ae6')
run('fix_git',['git','--no-optional-locks','-C',str(rt),'status','--short'])
run('fix_head',['git','--no-optional-locks','-C',str(rt),'rev-parse','HEAD'])
run('fix_diff',['git','--no-optional-locks','-C',str(rt),'show','8c7380c118ce7ca8a4ea4df53d753adc8fab0df2','--','robotwin/envs/vector_env.py'])
for rel in ('robotwin/envs/vector_env.py','robotwin/envs/_base_task.py'):
 p=rt/rel;lines=p.read_text().splitlines();print('SOURCE',str(p),'SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
 if 'vector_env' in rel: selected=set(range(len(lines)))
 else:
  selected=set()
  for i,l in enumerate(lines):
   if re.search(r'def (close_env|setup_demo|_init|__init__|_setup)|SapienRenderer|set_renderer|clear_cache|set_camera_shader_dir|denois',l):selected.update(range(max(0,i-4),min(len(lines),i+65)))
 print('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(selected)))
venv=Path('/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/lib/python3.11/site-packages')
print('SAPIEN_METADATA')
for pat in ('sapien-*.dist-info/METADATA','sapien-*.dist-info/direct_url.json'):
 for p in venv.glob(pat):print(str(p),'\n'.join(p.read_text().splitlines()[:22]))
for p in (venv/'sapien').rglob('*'):
 if p.is_file() and any(k in p.name.lower() for k in ('oidn','openimagedenoise','svulkan','libtbb')):print('NATIVE_LIB',str(p),p.stat().st_size)
PY
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import json,re,datetime,csv,os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1')
ea=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0});ea.Reload()
scalars={tag:[{'step':p.step+1,'value':p.value,'wall_time':p.wall_time} for p in ea.Scalars(tag)] for tag in ea.Tags()['scalars']}
log=(run/'runtime/driver.log').read_text(errors='replace')
resources=list(csv.DictReader((run/'runtime/resource.csv').open()))
ckpts=[]
for p in run.glob('*/checkpoints/global_step_*/actor/local_shard_checkpoint/checkpoint_rank_*.pt'):ckpts.append({'path':str(p),'bytes':p.stat().st_size})
pid=int((run/'runtime/wrapper.pid').read_text())
print('SIDNEY_JSON',json.dumps({'cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scalars':scalars,'resource':resources,'checkpoints':ckpts,'wrapper_alive':Path('/proc',str(pid)).exists(),'exit':(run/'runtime/exit_code.txt').read_text() if (run/'runtime/exit_code.txt').exists() else None,'last_progress':re.findall(r'Global Step:\s*\d+/100',log)[-1:],'last_rollout':[l for l in log.splitlines() if 'Generating Rollout Epochs:' in l][-1:],'fatal':{pat:len(re.findall(pat,log,re.I)) for pat in ('OIDN Error','pthread_key_create','Fatal Python error','Traceback','CUDA out of memory','OutOfMemoryError','non.?finite')}}))
PY
