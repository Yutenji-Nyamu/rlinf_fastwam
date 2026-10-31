"""One GPU, bounded tasks, exact owned cleanup, then its existing RLT fallback."""
import copy
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/pi05_distillation'))
from multitask_eval import cleanup, PYTHON
sys.path.insert(0,'/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops')
from common import proc, read, save


def main():
    D,gpu=Path(sys.argv[1]),int(sys.argv[2])
    plan=read(D/f'lane-g{gpu}-plan.json');O=Path(plan['output'])
    save(D/f'lane-g{gpu}-identity.json',proc(os.getpid()))
    results=[]; token=None; status={}; accepted=False; extra=None
    def stopped(*_):
        raise KeyboardInterrupt('lane stopped')
    signal.signal(signal.SIGTERM,stopped)
    try:
        while not (D/'registered.json').exists():
            if time.time()-plan['prepared_time']>1800:raise TimeoutError('handoff was not registered')
            time.sleep(.2)
        jobs=list(plan['jobs'])
        while jobs:
            job=jobs.pop(0);c=read(job['config']);out=Path(c['output'])
            # New writes use Home. On disk exhaustion release to the existing
            # RLT/no-save chain instead of repeatedly creating broken outputs.
            stat=os.statvfs(O)
            if stat.f_bavail*stat.f_frsize<2*2**30:
                status=dict(phase='storage_low',remaining=len(jobs)+1);break
            c['acceptance']=not accepted
            if extra:c['noise_extra_batch']=min(extra,3*c['num_envs'])
            save(Path(job['config']),c)
            token=f'{D.name}-g{gpu}-{Path(job["config"]).stem}'
            env=dict(os.environ,**read(O/f'environment-g{gpu}.json'),POLICY_EVAL_TOKEN=token)
            logpath=D/(Path(job['config']).stem+'.log')
            with logpath.open('x') as log:
                child=subprocess.Popen([PYTHON,'-u','-B',str(ROOT/'tools/pi05_u_video/run_batch.py'),job['config']],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                start=time.time();reason=None
                save(D/f'lane-g{gpu}-status.json',dict(phase='running',gpu=gpu,child=proc(child.pid),
                     config=job['config'],started=start,completed=len(results),remaining=len(jobs)))
                try:
                    while child.poll() is None:
                        time.sleep(1)
                        progress=out/'progress.json'
                        recent=progress.stat().st_mtime if progress.exists() else start
                        if time.time()-start>3600 or time.time()-recent>900:
                            reason='timeout';break
                finally:
                    cleanup(token)
                    child.wait(timeout=30)
            token=None
            valid=False;fatal=False
            if child.returncode==0 and reason is None:
                cpuenv=dict(env,CUDA_VISIBLE_DEVICES='');cpuenv.pop('DOJO_GPU_SCOPE',None)
                check=subprocess.run([PYTHON,'-B',str(ROOT/'tools/pi05_nstep_signal/validate.py'),job['config']],
                     cwd=ROOT,env=cpuenv,capture_output=True,text=True,timeout=300)
                (D/(logpath.stem+'-validation.log')).write_text(check.stdout+check.stderr)
                valid=check.returncode==0
                fatal=not valid
                reason='complete' if valid else 'validation_failed'
                if valid:
                    accepted=True
                    cap=read(out/'noise-capacity.json')
                    if cap:extra=cap['selected']['extra_batch']
            else:
                text=logpath.read_text(errors='replace')[-16000:]
                oom='out of memory' in text.lower()
                fatal=((out/'nstep_error.json').exists() or not (out/'progress.json').exists()) and not oom
                reason=reason or ('oom' if oom else 'task_error')
                if oom and c['num_envs']==32:
                    # Exactly the same 32 seeds, retried once in two B16 batches.
                    seedobj=read(c['env']['seeds_path']);seeds=seedobj[c['task']]['success_seeds']
                    retries=[]
                    for b in range(2):
                        cc=copy.deepcopy(c);name=out.name+f'_fallback{b}'
                        sf=O/'configs'/f'{name}_seeds.json'
                        save(sf,{c['task']:{'success_seeds':seeds[16*b:16*(b+1)]}})
                        cc.update(num_envs=16,batch=b,output=str(O/'batches'/name),noise_extra_batch=48)
                        cc['env'].update(total_num_envs=16,seeds_path=str(sf))
                        cc['env']['task_config']['save_path']=str(Path(cc['output'])/'env')
                        cc['env']['video_cfg']['video_base_dir']=str(Path(cc['output'])/'unused')
                        cp=O/'configs'/f'{name}.json';save(cp,cc);retries.append(dict(config=str(cp),task=c['task']))
                    jobs=retries+jobs
                if 'No space left on device' in text:fatal=True;reason='disk_full'
            row=dict(task=c['task'],config=job['config'],output=str(out),valid=valid,reason=reason,
                     rc=child.returncode,seconds=time.time()-start)
            if valid:row.update(read(out/'validation.json'))
            results.append(row)
            save(D/f'lane-g{gpu}-results.json',results)
            if fatal:
                status=dict(phase=reason,remaining=len(jobs));break
        else:status=dict(phase='complete',remaining=0)
    except BaseException:
        status=dict(phase='lane_error',error=traceback.format_exc())
        raise
    finally:
        if token:cleanup(token)
        status.update(time=time.time(),gpu=gpu,completed=len(results),
                      valid_episodes=sum(r.get('episodes',0)for r in results if r['valid']),
                      successes=sum(r.get('successes',0)for r in results if r['valid']))
        save(D/f'lane-g{gpu}-status.json',status)
        save(D/f'lane-g{gpu}-released.json',status)


if __name__=='__main__':
    main()
