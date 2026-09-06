#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python - <<'PY'
import csv,datetime,json,re,subprocess
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def cmd(args):
    return subprocess.run(args,capture_output=True,text=True,timeout=20).stdout.strip()
base=Path('/data/chenyiteng/results/rlinf-shenzhen')
runs={
'bc':base/'online-bc/pi0-adjust-bottle-smoke32x1-m4-gpu6-20260905-v6',
'sidney':base/'pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1',
}
result={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'runs':{}}
for name,run in runs.items():
    rt=run if name=='bc' else run/('runtime-resume100-to200' if (run/'runtime-resume100-to200').is_dir() else 'runtime')
    log=(rt/'driver.log').read_text(errors='replace')
    log=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',log)
    state={f:(rt/f).read_text().strip() if (rt/f).exists() else None for f in ['started_at.txt','exit_code.txt','finished_at.txt']}
    state['runtime_dir']=str(rt)
    state['progress']=[x[-1000:] for x in log.splitlines() if any(k in x for k in ['Global Step:', 'Generating Rollout Epochs:', 'Evaluating Epochs:', 'Train Epoch:', 'Saved checkpoint'])][-5:]
    state['errors']={k:log.count(k) for k in ['CUDA out of memory','OutOfMemoryError','Fatal Python error','RuntimeError:','AssertionError','Traceback','OIDN Error','pthread_key_create failed','ErrorInitializationFailed','ErrorOutOfHostMemory','Too many open files']}
    state['tail']=log.splitlines()[-14:]
    state['scalars']={}
    tb=run/'tensorboard'
    try:
        ea=EventAccumulator(str(tb),size_guidance={'scalars':0}); ea.Reload()
        for tag in ea.Tags().get('scalars',[]):
            if any(k in tag for k in ['success','query_records','bc/','actor/lr','grad_norm','time/step']):
                points=ea.Scalars(tag)
                state['scalars'][tag]=[{'step':p.step,'value':p.value} for p in points[-5:]]
    except Exception as e: state['tb_error']=str(e)
    checkpoints=[]
    roots=list(run.glob('*/checkpoints'))+([run/'checkpoints'] if (run/'checkpoints').exists() else [])
    for cr in roots:
        for step in cr.glob('global_step_*'):
            checkpoints.append({'path':str(step),'files':[{'path':str(p.relative_to(step)),'bytes':p.stat().st_size} for p in step.rglob('*') if p.is_file()]})
    state['checkpoints']=checkpoints
    if name=='bc':
        state['worker_rss_mib']={}
        for group,pid in set(re.findall(r'((?:Actor|Env|Rollout)Group\(rank=\d+\)) pid=(\d+)',log)):
            p=Path('/proc')/pid/'status'
            if p.exists():
                fields=dict(x.split(':',1) for x in p.read_text().splitlines() if ':' in x)
                if fields['Uid'].split()[0]=='1003':
                    state['worker_rss_mib'][group]={'pid':int(pid),'rss_mib':int(fields.get('VmRSS','0 kB').split()[0])/1024,
                    'open_fds':len(list((p.parent/'fd').iterdir())),
                    'nofile':[x for x in (p.parent/'limits').read_text().splitlines() if 'Max open files' in x]}
        if (run/'resource.csv').exists():
            rows=list(csv.DictReader((run/'resource.csv').open()))
            valid=[r for r in rows if r.get('gpu6_used_mib') and r['gpu6_used_mib'].isdigit()]
            if valid: state['resource']={'sampled_peak_gpu_mib':max(int(r['gpu6_used_mib']) for r in valid),'min_host_available_gib':min(int(r['host_mem_available_kib'])/1024**2 for r in valid),'max_memory_psi':max(float(r['mem_psi_some_avg10']) for r in valid),'latest':valid[-1]}
        state['archives']=[{'path':str(p.relative_to(run)),'bytes':p.stat().st_size} for p in (run/'success_data').rglob('batch_*.pt')]
    result['runs'][name]=state
result['gpu']=cmd(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
result['mem']=cmd(['free','-h'])
result['disk']=cmd(['df','-h','/data'])
result['cpu_swap']=cmd(['vmstat','1','2'])
result['preserved_processes']=cmd(['ps','-p','602620,321933,322685','-o','user,pid,etime,stat,comm'])
result['bc_git']=cmd(['git','-C','/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc','status','--short'])
result['bc_head']=cmd(['git','-C','/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc','rev-parse','HEAD'])
print(json.dumps(result,ensure_ascii=False,indent=2))
PY
