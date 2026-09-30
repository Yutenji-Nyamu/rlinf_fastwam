"""Observe one approved Dojo run; stop its exact stalled controller only.

The original pipeline remains the sole owner of cleanup and RLT restoration.
No GPU resets, Ray operations, task time limits, or training configuration edits.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time

STEP = re.compile(r'env(\d+) step:\s*(\d+)\s*/\s*(\d+)')
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
FATAL = re.compile(r'GPU crash (?:is detected|dump)|(?:VK_)?ERROR_DEVICE_LOST|CUDA error: an illegal memory access', re.I)
SAFE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,110}\Z')

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p, value):
    p=Path(p);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    with tmp.open('w') as stream: json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    tmp.chmod(0o600);os.replace(tmp,p)

def identity(pid, start=None):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    assert p.stat().st_uid==os.getuid() and fields[0] not in ('Z','X')
    if start is not None: assert int(fields[19])==start, 'PID start mismatch'
    raw=(p/'cmdline').read_bytes()
    return {'pid':pid,'start':int(fields[19]),'uid':p.stat().st_uid,'ppid':int(fields[1]),
            'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'command_sha256':hashlib.sha256(raw).hexdigest(),'argv':[os.fsdecode(x) for x in raw.split(b'\0') if x]}

def argument(argv, flag):
    assert argv.count(flag)==1
    return argv[argv.index(flag)+1]

def bound_run(run):
    """Fresh ownership, relationship, frozen-source and restore-path validation."""
    current=read(run/'pipeline-current.json')
    if current['phase']!='EVALUATING': return None
    active=read(run/'active-continuation.json')
    attempt=Path(active.get('attempt_dir',active.get('path'))).resolve()
    cycle=Path(active.get('cycle_dir',active.get('cycle'))).resolve()
    project=run.parent.parent.resolve()
    assert attempt.parent==run and cycle.is_relative_to(project)
    assert current['pid']==active['pid'] and Path(current['continuation']).resolve()==attempt
    assert not (attempt/'pipeline-final.json').exists() and not (cycle/'resumed-dispatched.json').exists()
    parent=identity(active['pid'],active.get('start'))
    child_spec=read(attempt/'dojo-controller-identity.json')
    child=identity(child_spec['pid'],child_spec['start'])
    assert child['ppid']==parent['pid'] and child['uid']==parent['uid']==os.getuid()
    config=Path(argument(child['argv'],'--config')).resolve()
    assert config==Path(argument(parent['argv'],'--config')).resolve() and config.is_relative_to(project)
    assert Path(argument(parent['argv'],'--cycle-dir')).resolve()==cycle
    cfg=read(config);plan=read(run/'plan.json')
    assert cfg['uid']==os.getuid() and cfg['run_id']==run.name and Path(cfg['project']).resolve()==project
    assert plan['config']==cfg
    script=next(Path(x) for x in child['argv'] if x.endswith('/dojo_sweep.py'))
    assert script.resolve().is_relative_to(project) and sha(script)==plan['controller_sha256']['dojo_sweep.py']
    wrapper=next(Path(x) for x in parent['argv'] if x.endswith('/continue_pipeline.py') or x.endswith('/formal_pipeline.py'))
    assert wrapper.resolve().is_relative_to(project)
    text=wrapper.read_text()
    assert all(v in text for v in ("finally:","--cleanup-only","processes_clear","gpus_released","--release-receipt"))
    stop=read(cycle/'rlt-stopped.json');rp=read(cycle/'plan.json')
    assert set(stop['runs'])==set(rp['runs'])=={'gpu4','gpu5','gpu6','gpu7'}
    helper=cycle/('rlt_cycle_sz3.py' if 'sz3' in run.name else 'rlt_cycle.py')
    assert sha(helper)==rp['script_sha256']
    records=[read(p) for p in run.glob('controller-*.json')]
    meta=next(r for r in records if r['pid']==child['pid'] and int(r['stat'].rsplit(')',1)[1].split()[19])==child['start'])
    return {'parent':parent,'controller':child,'attempt':str(attempt),'cycle':str(cycle),'token':str(meta['attempt']),
            'repo':cfg['repo'],'started_at':meta['started_at']}

class Progress:
    def __init__(self, now):
        self.last=now;self.steps={};self.count=None;self.fatal=False
        self.offset=0;self.inode=None;self.carry=''
    def observe(self, text, count, now):
        text=ANSI.sub('',text);self.fatal=self.fatal or bool(FATAL.search(text))
        advanced=False
        for env,step,total in STEP.findall(text):
            item=(int(step),int(total))
            if self.steps.get(env)!=item: advanced=True
            self.steps[env]=item
        if count is not None and (self.count is None or count>self.count):
            if count>0: advanced=True
            self.count=count
        if advanced: self.last=now
        return advanced
    def poll(self, log, count, now):
        text=''
        if log.exists():
            st=log.stat()
            if self.inode!=st.st_ino or st.st_size<self.offset:
                self.offset=max(0,st.st_size-1048576);self.inode=st.st_ino;self.carry=''
            # Bounded incremental reading; warning writes never reset the clock.
            with log.open('rb') as stream:
                stream.seek(self.offset);data=stream.read(1048576);self.offset=stream.tell()
            text=self.carry+data.decode(errors='replace')
            split=max(text.rfind('\n'),text.rfind('\r'))
            self.carry=text[split+1:][-4096:];text=text[:split+1]
        self.observe(text,count,now)
    def failure(self, now, fatal_stall=300, stall=1800):
        elapsed=now-self.last
        if self.fatal and elapsed>=fatal_stall: return 'gpu_fatal_and_no_progress'
        if elapsed>=stall: return 'no_action_or_episode_progress'
        return None

def task_rows(run, bound):
    rows=[]
    for file in sorted(run.glob('seed*/summary.json')):
        summary=read(file)
        for task,row in summary['results'].items():
            if row['status']!='RUNNING': continue
            assert SAFE.fullmatch(task) and row['seed'] in (0,1,2) and row['worker'] in range(8)
            log=run/f"seed{row['seed']}/worker{row['worker']}/tasks/{task}/stdout-{bound['token']}.log"
            if row.get('log_path'):
                selected=Path(row['log_path']).resolve()
                assert selected.is_relative_to((run/f"seed{row['seed']}/worker{row['worker']}/tasks/{task}").resolve())
                log=selected
            # Ignore stale RUNNING rows left by an older controller attempt.
            if row.get('started_at',0)<bound['started_at']: continue
            result_dir=Path(bound['repo'])/'eval_result/RoboDojo'/task
            results=list(result_dir.glob(f"*/*/{row['seed']}_*/{row['run_id']}/_result.json"))
            assert len(results)<=1
            count=None
            if results:
                try:
                    value=read(results[0]);count=int(value['eval_time'])
                    if len(value['details'])!=count: count=None
                except (json.JSONDecodeError,FileNotFoundError): pass
            # COMPLETE workers were excluded above. A RUNNING client can also
            # hang during shutdown after full budget, so retain it here.
            rows.append((f"{row['seed']}/{task}/retry{row.get('recovery_attempt',0)}",log,count,row['worker']))
    return rows

def signal_bound(run,bound,sig):
    fresh=bound_run(run)
    assert fresh is not None and fresh['parent']==bound['parent'] and fresh['controller']==bound['controller']
    fd=os.pidfd_open(bound['controller']['pid'])
    try:
        assert identity(bound['controller']['pid'],bound['controller']['start'])==bound['controller']
        signal.pidfd_send_signal(fd,sig)
    finally: os.close(fd)

def run_watch(run,interval=30):
    run=run.resolve();assert SAFE.fullmatch(run.name) and run.stat().st_uid==os.getuid()
    folder=run/'hang-watchdog';folder.mkdir(exist_ok=True)
    with (folder/'watchdog.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        states={};active_key=None;stop=False
        def terminate(_sig,_frame):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
        atomic(folder/'identity.json',identity(os.getpid()))
        while not stop:
            now=time.monotonic();record={'time':time.time(),'action':'none','uid':os.getuid()}
            try:
                bound=bound_run(run)
                if bound is None:
                    states={};active_key=None;record['state']='IDLE_RLT_OR_FINISHED'
                else:
                    key=f"{bound['controller']['boot']}-{bound['controller']['pid']}-{bound['controller']['start']}"
                    if key!=active_key: states={};active_key=key
                    record.update(state='WATCHING',controller=bound['controller'],tasks=[])
                    rows=task_rows(run,bound);present={row[0] for row in rows}
                    states={k:v for k,v in states.items() if k in present}
                    triggered=None
                    for task,log,count,worker in rows:
                        progress=states.setdefault(task,Progress(now));progress.poll(log,count,now)
                        reason=progress.failure(now)
                        detail={'task':task,'worker':worker,'episodes':count,'stalled_seconds':round(now-progress.last,1),
                                'fatal_seen':progress.fatal,'reason':reason}
                        record['tasks'].append(detail)
                        if reason: triggered=detail
                    receipt=folder/(key+'-term.json')
                    if triggered and not receipt.exists():
                        # Durable intent prevents ambiguous repeated sends.
                        intent=folder/(key+'-intent.json')
                        with intent.open('x') as stream: json.dump({'time':time.time(),'bound':bound,'trigger':triggered},stream,indent=2)
                        signal_bound(run,bound,signal.SIGTERM)
                        atomic(receipt,{'time':time.time(),'signal':'SIGTERM','bound':bound,'trigger':triggered})
                        record.update(state='STOP_REQUESTED',action='SIGTERM exact Dojo controller')
                    elif receipt.exists():
                        record['state']='WAITING_FOR_ORIGINAL_PIPELINE_RETURN'
                        if time.time()-read(receipt)['time']>300:
                            record['state']='CONTROLLER_STOP_NEEDS_ATTENTION'
            except Exception as exc:
                record.update(state='NEEDS_ATTENTION',error=repr(exc),action='none; ownership or state validation failed')
            atomic(folder/'current.json',record)
            deadline=now+interval
            while not stop and time.monotonic()<deadline: time.sleep(min(1,deadline-time.monotonic()))
        atomic(folder/'stopped.json',{'time':time.time(),'training_action':'none'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run_watch(args.run)
