"""Continue at the existing complete endpoint while retaining the same GPU lease.

The lease's nonblocking operation lock fences the old owner's return dispatch.
The old trainer exits normally and writes its real exit/terminal receipts; the
old CPU owner then exits at the claimed return lock. This owner checks those
receipts and either launches the reviewed continuation or returns original RLT.
It never signals a running trainer, edits its source, or resets shared Ray.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,signal,subprocess,time,traceback
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def module(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def save(p,v,exclusive=False):
    p=Path(p)
    if exclusive:
        with p.open('x') as f:json.dump(v,f,indent=2)
    else:
        t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def changed_config(old,oldrun,newrun,start,target):
    import copy
    assert 0<start<target
    def rep(x):
        if isinstance(x,dict):return {k:rep(v) for k,v in x.items()}
        if isinstance(x,list):return [rep(v) for v in x]
        return x.replace(str(oldrun),str(newrun)).replace(oldrun.name,newrun.name) if isinstance(x,str) else x
    cfg=rep(copy.deepcopy(old));cfg['runner'].update(max_steps=target,max_epochs=target,
        resume_dir=str(oldrun/oldrun.name/'checkpoints'/f'global_step_{start}'))
    return cfg
def decision(terminal,finished,operation_id,gpu):
    assert terminal['operation_id']==operation_id and terminal['gpu']==gpu
    assert terminal['released'] is True,'No proven release'
    assert terminal['decision'] in ('COMPLETE','FAILED')
    return 'continue' if terminal['decision']=='COMPLETE' and finished and finished['exit_code']==0 else 'return'
def checkpoint(plan,extension,base):
    run=Path(extension['original_run']);step=extension['from_step'];cp=run/run.name/'checkpoints'/f'global_step_{step}'
    if plan['lane']=='rlt':
        value=base._checkpoint(run);assert value['step']==step;return value
    import torch
    small=cp/'actor/online_bc/rank_0'
    paths=[cp/'actor/model_state_dict/full_weights.pt',cp/'actor/local_shard_checkpoint/checkpoint_rank_0.pt',
           small/'learner.pt',small/'ugrow.pt',small/'success_replay.pt']
    assert all(p.is_file() and p.stat().st_size>0 for p in paths)
    assert not (small/'dvac.pt').exists()
    learner=torch.load(small/'learner.pt',map_location='cpu',weights_only=True)
    calibration=torch.load(small/'ugrow.pt',map_location='cpu',weights_only=True)
    assert learner['update_step']>0 and calibration
    return {'path':str(cp),'step':step,'update_step':learner['update_step'],
        'files':{str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths},
        'small_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths[2:4]}}
def cpu_env():
    env={k:v for k,v in os.environ.items() if k not in ('CUDA_VISIBLE_DEVICES','LD_PRELOAD','PYTHONPATH','RLINF_OPENDW_GPU_SCOPE_MANIFEST')}
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1');return env
def prune_bc(plan,extension,rt,common,stage):
    if plan['lane']!='bc':return
    assert extension['retention_authorized']=='latest2+resume+final; only this BC; preserve logs'
    complete=[];pinned=[Path(k).resolve() for k in plan['pins']]
    for value in (extension['original_run'],plan['runs']['formal']['run']):
        root=Path(value);directory=root/root.name/'checkpoints'
        m=rt.scalars(root);rows=m.get('train/ugrow/round',[])
        if not rows:continue
        completed=int(rows[-1]['value'])
        for cp in directory.glob('global_step_*'):
            step=int(cp.name.rsplit('_',1)[1])
            if step>completed:continue  # Native scalar is logged after synchronous save.
            assert not cp.is_symlink() and cp.resolve().parent==directory.resolve()
            required=['actor/model_state_dict/full_weights.pt','actor/local_shard_checkpoint/checkpoint_rank_0.pt',
                'actor/online_bc/rank_0/success_replay.pt','actor/online_bc/rank_0/learner.pt','actor/online_bc/rank_0/ugrow.pt']
            assert all((cp/f).is_file() and (cp/f).stat().st_size>0 for f in required)
            complete.append((step,cp,completed))
    complete.sort(key=lambda v:v[0]);keep={p for _,p,_ in complete[-2:]}
    keep|={p for s,p,_ in complete if s in (extension['from_step'],extension['total_steps'])}
    for step,cp,logged in complete:
        if cp in keep:continue
        assert all(not q.is_relative_to(cp.resolve()) for q in pinned),'Checkpoint is pinned'
        files=list(cp.rglob('*'));assert files and all(not f.is_symlink() and f.stat().st_uid==1003 for f in [cp,*files])
        # Check every process owned by this user for open references to this exact CP.
        for proc in Path('/proc').iterdir():
            try:
                if not proc.name.isdigit() or proc.stat().st_uid!=1003:continue
                for fd in (proc/'fd').iterdir():
                    try:target=os.readlink(fd)
                    except (FileNotFoundError,PermissionError,OSError):continue
                    assert not target.startswith(str(cp)+'/'),'Checkpoint still open: '+str(cp)
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        evidence={'time':time.time(),'checkpoint':str(cp),'step':step,'logged_complete_round':logged,
            'retained':[str(p) for p in sorted(keep)],'files':[{'path':str(f.relative_to(cp)),'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in files if f.is_file()]}
        receipt=Path(stage)/f'prune-step-{step}.json';assert not receipt.exists()
        save(receipt,evidence,True)
        for f in files:
            if f.is_file():f.unlink()
        for f in sorted((f for f in files if f.is_dir()),key=lambda p:len(p.parts),reverse=True):f.rmdir()
        cp.rmdir();evidence.update(deleted=True,finished_at=time.time());save(receipt,evidence)
def run(stage):
    stage=Path(stage);p=read(stage/'plan.json');e=read(stage/'extension.json')
    rt=module(p['ops'],'runtime');op,b=rt.configure(stage);op.checked()
    base=module(Path(p['ops']).with_name('rlt_lease.py'),'lease')
    own=(stage/'extension.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    oldstage=Path(e['original_stage']);oldrt=module(read(oldstage/'plan.json')['ops'],'oldruntime');oldop,_=oldrt.configure(oldstage)
    old=oldop.checked();ld=Path(p['lease_dir']);me=b.proc(os.getpid());save(stage/'extension-identity.json',me,True)
    state={'time':time.time(),'identity':me,'phase':'STARTING','from_step':e['from_step'],'total_steps':e['total_steps']}
    stop=False;dispatched=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    def fallback(terminal):
        for _ in range(120):
            v=base.restore(ld,terminal);save(stage/'fallback-status.json',v)
            if v['first_round_verified']:return
            if not v['alive']:raise RuntimeError('Fallback RLT exited')
            time.sleep(15)
        raise RuntimeError('Fallback first round not yet verified')
    try:
        with base._lock(ld):
            assert b.same(e['original_owner']) and b.same(e['original_driver'])
            assert not (oldstage/'terminal.json').exists() and not (ld/'return-attempt.json').exists()
            assert not (ld/'return-dispatched.json').exists()
            save(stage/'armed.json',{'time':time.time(),'identity':me,'lease':str(ld),'guard':'exclusive operation.lock held',
                'original_owner':e['original_owner'],'original_driver':e['original_driver'],'from_step':e['from_step'],'total_steps':e['total_steps']},True)
            while b.same(e['original_owner']):
                if stop:
                    state.update(phase='CANCELLED',time=time.time());save(stage/'extension-status.json',state);return
                state.update(phase='WAITING_ORIGINAL_ENDPOINT',time=time.time(),original_driver_alive=b.same(e['original_driver']))
                if p['lane']=='bc':prune_bc(p,e,rt,b,stage)
                save(stage/'extension-status.json',state);time.sleep(15)
            terminal=read(oldstage/'terminal.json');finished_path=Path(e['original_run'])/'runtime/finished.json'
            finished=read(finished_path) if finished_path.exists() else None
            action=decision(terminal,finished,p['operation_id'],p['gpu'])
            assert not b.same(e['original_driver'])
            scope=oldrt.scope_state(oldop,b,old,old['runs']['formal'],e['original_driver'])
            assert scope['released'] and not scope['outside']
            assert all(not b.same(q) for r in terminal['runs'] for q in [r['identity'],*r.get('processes',[])])
            if action=='continue':
                op.checked();cp=checkpoint(p,e,base)
                import yaml
                cfg=yaml.safe_load((Path(p['runs']['formal']['run'])/'runtime/resolved.yaml').read_text())
                assert cfg['runner']['resume_dir']==cp['path'] and cfg['runner']['max_steps']==cfg['runner']['max_epochs']==e['total_steps']
                save(stage/'resume-verified.json',{'time':time.time(),'checkpoint':cp,'original_finished':finished,'original_terminal':str(oldstage/'terminal.json')},True)
            save(stage/'handoff.json',{'time':time.time(),'action':action,'original_owner_exited':True,'original_driver_exit':finished,
                'scope_released':True,'note':'Original return was fenced by the extension lease claim.'},True)
        if action=='return':
            fallback(oldstage/'terminal.json');state['phase']='ORIGINAL_FAILED_RLT_RETURNED'
        else:
            assert not (stage/'owner-launch.json').exists() and not (stage/'launch-authorized.json').exists()
            save(stage/'launch-authorized.json',{'time':time.time(),'operation_id':p['operation_id'],'gpu':p['gpu'],'lease_dir':p['lease_dir']},True)
            with (stage/'owner.log').open('x') as f:
                child=subprocess.Popen([p['python'],'-u','-B',p['ops'],'--stage',str(stage),'owner'],cwd=p['repo'],env=cpu_env(),
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            dispatched=True;record={'time':time.time(),'identity':b.proc(child.pid),'plan':str(stage/'plan.json')}
            save(stage/'owner-launch.json',record,True);state.update(phase='CONTINUATION_DISPATCHED',continuation=record)
            while p['lane']=='bc' and b.same(record['identity']):
                prune_bc(p,e,rt,b,stage)
                state.update(phase='CONTINUATION_RUNNING_RETENTION',time=time.time());save(stage/'extension-status.json',state);time.sleep(30)
            if p['lane']=='bc':
                prune_bc(p,e,rt,b,stage);state['phase']='CONTINUATION_OWNER_FINISHED'
    except BaseException:
        state.update(phase='NEEDS_ATTENTION',error=traceback.format_exc())
        terminal=oldstage/'terminal.json'
        if not dispatched and not b.same(e['original_owner']) and terminal.is_file() and read(terminal).get('released'):
            try:fallback(terminal);state['fallback_completed']=True
            except BaseException:state['fallback_error']=traceback.format_exc()
    finally:
        state['time']=time.time();save(stage/'extension-status.json',state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);run(parser.parse_args().stage)
