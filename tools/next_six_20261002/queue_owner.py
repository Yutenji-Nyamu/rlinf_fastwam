"""One task: proven Stage1, then independent, receipt-gated GPU cutovers.

No borrowed workload is signalled; frozen owners finish their existing return.
All process cleanup is delegated to the existing scoped RLT operations.
"""
import argparse, copy, fcntl, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import ops

ROOT=Path('/data/chenyiteng')
WATCH=ROOT/'deployment-20260927/rlt-six-task-watch/plan.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+f'.next6-{os.getpid()}.tmp')
    with temp.open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def gate(plan,role):
    spec=plan['gates'][role]
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==plan['boot_id'], 'Host rebooted; revalidate queue'
    assert sha(spec['cycle_plan'])==spec['cycle_sha256'], 'Frozen return plan changed'
    for path,digest in spec.get('anchors',{}).items():
        assert sha(path)==digest, 'Active borrowing owner changed: '+path
    if spec['kind']=='lane':
        state=ops.read(spec['state_file'])['lanes'][str(spec['gpu'])]['state']
        if state not in ('RLT_RESTORED','NEEDS_ATTENTION'):return None
    else:
        if not Path(spec['final']).is_file() or ops.same(spec['owner']):return None
    receipt=None
    for path in spec['receipts']:
        if Path(path).is_file():
            value=ops.read(path)
            value=value.get('status',value)
            if value.get('all_first_rounds_verified') is True:
                receipt=value;break
    if receipt is None:
        return None
    cycle=ops.read(spec['cycle_plan'])
    assert receipt['cycle_id']==cycle['cycle_id'], 'Return receipt is for another cycle'
    key='gpu'+str(spec['gpu'])
    proof=receipt['runs'][key]
    assert proof['first_round_verified'] is True
    target=next(x for x in plan['old_runs'] if x['gpus']==[spec['gpu']])
    assert target['run']==cycle['runs'][key]['new_run'] and target['namespace']==cycle['runs'][key]['namespace']
    identity=ops.read(Path(target['run'])/'runtime/driver-identity.json')
    old=ops.normalized(proof['resume_identity'])
    assert all(identity[k]==old[k] for k in ('pid','uid','start')), 'Returned driver identity changed'
    assert identity['namespace']==target['namespace'] and identity['uid']==plan['uid']
    routes=[x for x in ops.read(WATCH)['runs'].values() if x['gpus']==target['gpus']]
    assert len(routes)==1 and all(routes[0][k]==target[k] for k in ('run','namespace')), 'Watch no longer names the frozen old run'
    return {**target,'identity':identity}

def update_watch(plan,role):
    row=plan['runs'][role];old=next(x for x in plan['old_runs'] if x['gpus']==row['gpus'])
    with (WATCH.parent/'route.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=ops.read(WATCH);after=copy.deepcopy(before)
        found=[(key,x) for key,x in after['runs'].items() if x['gpus']==row['gpus']]
        assert len(found)==1
        key,target=found[0]
        assert target['run'] in (old['run'],row['run'])
        if target['run']==row['run']:return
        ops.save(ops.ST/f'{role}-watch-before.json',before)
        target.update(run=row['run'],namespace=row['namespace'])
        # Preserve the established watch schema and unrelated GPU routes.
        if 'task' in target:target['task']=plan['task']
        atomic(WATCH,after)
        assert ops.read(WATCH)['runs'][key]==target
        ops.save(ops.ST/f'{role}-watch-updated.json',{'time':ops.now(),'key':key,'target':target})

def first_collection(row):
    sys.modules.setdefault('tensorflow',None)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run=Path(row['run'])
    for directory in [run/'tensorboard',run/run.name/'tensorboard']:
        if not directory.is_dir():continue
        acc=EventAccumulator(str(directory),size_guidance={'scalars':1});acc.Reload()
        if 'env/success_once' in acc.Tags()['scalars']:
            last=acc.Scalars('env/success_once')[-1]
            return {'step':last.step,'success':last.value,'wall_time':last.wall_time}
    return None

def stage1_gate(plan):
    spec=plan.get('stage1_gate')
    if spec:
        if ops.same(spec['owner']):return False,'WAITING_OFFICIAL_EVAL'
        if not any(Path(p).is_file() for p in spec['terminal_files']):return False,'WAITING_OFFICIAL_EVAL_RECEIPT'
    if ops.gpu_pids(plan['runs']['stage1-full']['gpus']):return False,'WAITING_STAGE1_GPU'
    return True,'READY'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    args=parser.parse_args();ops.ST=Path(args.stage).resolve();plan=ops.checked()
    lock=(ops.ST/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ops.ST/'owner-identity.json').exists(),'Never replay an existing owner'
    ops.save(ops.ST/'owner-identity.json',{**ops.proc(os.getpid()),'time':ops.now(),'boot_id':plan['boot_id']})
    state={'time':ops.now(),'task':plan['task'],'stage1':'WAITING_DATA','roles':{k:'WAITING_STAGE1' for k in ['clean','combo']}}
    children={};stage_child=None;stage_done=False;first=set();failures={}
    def publish():
        state['time']=ops.now();atomic(ops.ST/'queue-status.json',state)
    publish()
    try:
        while True:
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==plan['boot_id']
            if not stage_done:
                if stage_child is None:
                    if Path(plan['dataset_marker']).is_file():
                        ready,reason=stage1_gate(plan);state['stage1']=reason
                        if ready:
                            stage_child=ops.launch('stage1-full');state['stage1']='TRAINING'
                    elif Path(plan['data_error']).is_file():raise RuntimeError('Official data preparation failed; old RLT retained')
                elif stage_child.poll() is not None:
                    ops.wait_stage1(stage_child);stage_done=True;state['stage1']='COMPLETE'
            if stage_done:
                for role in ['clean','combo']:
                    if role in children or role in failures or state['roles'][role]=='COMPLETE':continue
                    try:
                        target=gate(plan,role)
                        if target is None:state['roles'][role]='WAITING_BORROW_RETURN';continue
                        # Bind one exact returned process. Proven stop_old still validates
                        # namespace, Ray jobs, UID, PID start time and all GPU processes.
                        ops.save(ops.ST/f'{role}-cutover-bound.json',{'time':ops.now(),'target':target})
                        ops.stop_old([target],role+'-')
                        child=ops.launch(role);children[role]=child;state['roles'][role]='STARTING'
                        ops.save(ops.ST/f'{role}-dispatched.json',{'time':ops.now(),'identity':ops.proc(child.pid)})
                        for _ in range(60):
                            if (Path(plan['runs'][role]['run'])/'runtime/driver-identity.json').is_file():break
                            if child.poll() is not None:raise RuntimeError(f'{role} driver exited on startup')
                            time.sleep(1)
                        else:raise RuntimeError(f'{role} driver identity missing')
                        update_watch(plan,role)
                    except Exception:
                        failures[role]=traceback.format_exc();state['roles'][role]='NEEDS_ATTENTION'
                        ops.save(ops.ST/f'{role}-failure.json',{'time':ops.now(),'error':failures[role]})
                for role,child in list(children.items()):
                    code=child.poll()
                    if code is not None:
                        ops.finished(role,code);state['roles'][role]='COMPLETE' if code==0 else 'FAILED'
                        if code:failures[role]=f'exit {code}'
                        del children[role]
                    elif role not in first:
                        result=first_collection(plan['runs'][role])
                        if result:
                            ops.save(ops.ST/f'{role}-first-round.json',{'time':ops.now(),'metrics':result})
                            first.add(role);state['roles'][role]='TRAINING'
            publish()
            if stage_done and all(state['roles'][r] in ['COMPLETE','FAILED','NEEDS_ATTENTION'] for r in ['clean','combo']):break
            time.sleep(30)
        ops.save(ops.ST/'queue-finished.json',{'time':ops.now(),'failures':failures})
    except BaseException:
        state['error']=traceback.format_exc();state['stage1']='FAILED' if not stage_done else state['stage1'];publish()
        ops.save(ops.ST/'queue-failure.json',{'time':ops.now(),'error':state['error']})
        raise

if __name__=='__main__':main()
