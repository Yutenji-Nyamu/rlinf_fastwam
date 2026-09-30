"""Bounded same-task recovery; no benchmark, physics or policy changes."""
import json
from pathlib import Path
import re
import time

ANSI=re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
STEP=re.compile(r'env(\d+) step:\s*(\d+)\s*/\s*(\d+)')
FATAL=re.compile(r'GPU crash (?:is detected|dump)|(?:VK_)?ERROR_DEVICE_LOST|'
                 r'cudaError(?:IllegalAddress|MemoryAllocation)|'
                 r'PhysX Internal CUDA error|CUDA error 700',re.I)

class Progress:
    def __init__(self,now):
        self.last=now;self.steps={};self.count=None;self.fatal=False
        self.offset=0;self.carry=''
    def observe(self,text,count,now):
        text=ANSI.sub('',text);self.fatal |= bool(FATAL.search(text))
        changed=False
        for env,step,total in STEP.findall(text):
            value=(int(step),int(total))
            changed |= self.steps.get(env)!=value;self.steps[env]=value
        if count is not None:
            changed |= self.count is not None and count>self.count
            self.count=count
        if changed:self.last=now
    def poll(self,log,result,now):
        text=''
        if log.exists():
            with log.open('rb') as f:
                f.seek(self.offset);data=f.read(1048576);self.offset=f.tell()
            text=self.carry+data.decode(errors='replace')
            split=max(text.rfind('\n'),text.rfind('\r'))
            self.carry=text[split+1:][-4096:];text=text[:split+1]
        count=None
        try:
            data=json.loads(result.read_text());n=data['eval_time']
            if type(n) is int and len(data['details'])==n:count=n
        except (OSError,ValueError,KeyError,TypeError):pass
        self.observe(text,count,now)
    def failure(self,now):
        idle=now-self.last
        if self.fatal and idle>=120:return 'fatal_without_progress_120s'
        if idle>=1800:return 'no_action_or_episode_progress_1800s'
        return None

def run_client(sweep,seed,task,worker,gpu,command,work_dir,server,server_id,server_command,server_env):
    # Import here: resume_results uses the controller's strict result validator.
    from resume_results import prepare_result
    from dojo_sweep import result_check
    run_id=sweep.task_id(seed,task);result=sweep.result_path(seed,task)
    expected=sweep.plan['budgets'][task]
    total_started=time.time()
    for attempt in range(3):
        if sweep.stop.is_set():break
        sweep.verify_sources()
        directory=work_dir/'tasks'/task/f'recovery-{sweep.attempt}-{attempt}'
        directory.mkdir(parents=True,exist_ok=False)
        row=prepare_result(sweep.repo,task,seed,run_id,expected,directory/'preserved',apply=True)
        sweep.event('resume_validated',task=task,seed=seed,recovery_attempt=attempt,detail=row)
        if row['action']=='skip_complete':
            sweep.update(seed,task,status='COMPLETE',result=result_check(result,expected));return server
        # Every generation has a distinct log and durable process receipt.
        log=directory/f'stdout-{sweep.attempt}.log'
        sweep.update(seed,task,status='RUNNING',worker=worker,gpu=gpu,started_at=time.time(),
                     recovery_attempt=attempt,log_path=str(log))
        client=None;reason=None
        progress=Progress(time.monotonic())
        try:
            client=sweep.launch(command,run_id,'client',gpu,sweep.cfg['sim_env'],directory)
            while client.poll() is None and not sweep.stop.wait(2):
                if server.poll() is not None:
                    reason='policy_server_exited';break
                progress.poll(log,result,time.monotonic())
                reason=progress.failure(time.monotonic())
                if reason:break
        finally:
            receipt=sweep.guard.cleanup(run_id,'task_attempt_finished_or_stalled')
            if client is not None:client.wait(timeout=10)
        check=result_check(result,expected)
        if check['complete']:
            sweep.update(seed,task,status='COMPLETE',result=check,exit_code=client.returncode,
                         wall_seconds=time.time()-total_started,cleanup_receipt=receipt)
            sweep.event('task_finished',task=task,seed=seed,worker=worker,status='COMPLETE',
                        complete_episodes=check['eval_time'],expected=expected)
            return server
        if sweep.stop.is_set():
            sweep.update(seed,task,status='INTERRUPTED',result=check,cleanup_receipt=receipt);return server
        reason=reason or f'incomplete_client_exit_{client.returncode}'
        sweep.update(seed,task,status='RETRYING',result=check,retry_reason=reason,cleanup_receipt=receipt)
        sweep.event('task_retry',task=task,seed=seed,worker=worker,reason=reason,
                    finished_attempt=attempt,completed_episodes=check.get('eval_time'),max_attempts=3)
        if attempt==2:raise RuntimeError(f'{task}/s{seed}: three incomplete attempts; last={reason}')
        # Restart the matching policy process too, discarding a possibly broken
        # CUDA context or WebSocket; all completed native results remain intact.
        sweep.guard.cleanup(server_id,'before_same_task_retry')
        server.wait(timeout=10)
        if sweep.stop.wait(10):return server
        server=sweep.launch(server_command,server_id,'server',gpu,server_env,
                            work_dir/'server'/f'recovery-{sweep.attempt}-{attempt+1}')
        sweep.wait_server(server,int(sweep.plan['groups'][worker]['port']))
    return server
