"""One assigned GPU: smoke, formal, then exactly one original RLT return."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

BASE_OPS='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rlt-next-six-sz1-20261002/tools/next_six_20261002/ops.py'
COMMON='/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-after-dojo-n25-sz1-20261004/tools/rlt_after_dojo_n25_20261004/owner.py'
COMMON_SHA='8e21677d0e3b0d58b810edab574d2903a54161be41cadff716b23f8f96d74029'

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def read(path):return json.loads(Path(path).read_text())

def save(path,value,exclusive=False):
    path=Path(path)
    if exclusive:
        with path.open('x') as f:json.dump(value,f,indent=2)
    else:
        temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def configure(stage):
    p=read(Path(stage)/'plan.json')
    assert hashlib.sha256(Path(COMMON).read_bytes()).hexdigest()==COMMON_SHA
    assert hashlib.sha256(Path(BASE_OPS).read_bytes()).hexdigest()==p['pins'][BASE_OPS]
    op=module(BASE_OPS,'scoped_ops');op.ST=Path(stage)
    op.__file__=__file__
    def checked():
        p=read(op.ST/'plan.json')
        assert os.getuid()==p['uid']==1003
        assert p['gpu'] in (4,5)
        assert Path(p['ops']).resolve()==Path(__file__).resolve()
        assert set(p['runs'])=={'smoke','formal'}
        assert all(row['gpus']==[p['gpu']] for row in p['runs'].values())
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==p['boot_id']
        assert subprocess.check_output(['git','-C',p['repo'],'rev-parse','HEAD'],text=True).strip()==p['head']
        assert not subprocess.check_output(['git','-C',p['repo'],'status','--porcelain'],text=True).strip()
        b=module(COMMON,'checked_common')
        for name,digest in p['pins'].items():assert b.sha(name)==digest,name
        return p
    op.checked=checked
    return op,module(COMMON,'common')

def scalars(run):
    sys.modules.setdefault('tensorflow',None)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out={}
    for p in (Path(run)/'tensorboard',Path(run)/Path(run).name/'tensorboard'):
        if p.is_dir():
            a=EventAccumulator(str(p),size_guidance={'scalars':0});a.Reload()
            for k in a.Tags()['scalars']:
                out[k]=[{'step':v.step,'value':v.value,'time':v.wall_time} for v in a.Scalars(k)]
    return out

def scope_state(op,b,p,row,identity):
    actors=op.validate_scoped_actors(p,op.active_actors(p,row['namespace']))
    roots={a['pid'] for a in actors if a.get('pid')}
    if identity and b.same(identity):roots.add(identity['pid'])
    tree=op.process_tree(roots,1003)
    gpu=b.gpu_snapshot()
    outside=[g for g in gpu if g['gpu']!=p['gpu'] and any(r['pid'] in tree for r in g['processes'])]
    return {'actors':actors,'tree':list(tree.values()),'outside':outside,'gpu':gpu[p['gpu']],
            'released':not actors and not tree and b.empty_target(gpu[p['gpu']],p['gpu_uuid'])}

def smoke_gate(p,row):
    x=scalars(row['run'])
    assert x,'No smoke metrics'
    bad=[k for k,v in x.items() if k.startswith('train/') and any(not math.isfinite(q['value']) for q in v)]
    assert not bad,bad
    updates={k:v[-1] for k,v in x.items() if 'update_step' in k and v}
    assert any(v['value']>0 for v in updates.values()),updates
    ugrow={k:v for k,v in x.items() if 'ugrow' in k}
    assert ugrow,'No explicit U telemetry'
    effect={k:v for k,v in ugrow.items() if 'weight_nonunit_fraction' in k or 'w_nonuniform_count' in k}
    assert any(q['value']>0 for values in effect.values() for q in values),'No nontrivial U weighting in smoke'
    return {'time':time.time(),'updates':updates,'ugrow':ugrow,'finite_train_metrics':True}

def owner(stage):
    import fcntl
    op,b=configure(stage);p=op.checked();stage=Path(stage)
    lock=(stage/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    me=b.proc(os.getpid());save(stage/'owner-identity.json',me,True)
    lease=module(Path(__file__).with_name('rlt_lease.py'),'lease')
    state={'time':time.time(),'owner':me,'stage':'READY','gpu':p['gpu'],'runs':[]}
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    # The lease is prepared and original RLT is released before this owner starts.
    authorization=read(stage/'launch-authorized.json')
    assert authorization['gpu']==p['gpu'] and authorization['operation_id']==p['operation_id']
    assert authorization['lease_dir']==p['lease_dir']
    child=None;row=None;identity=None
    try:
        for key in ('smoke','formal'):
            if stopping:raise RuntimeError('Owner stopped before '+key)
            p=op.checked();row=p['runs'][key]
            assert b.empty_target(b.gpu_snapshot()[p['gpu']],p['gpu_uuid'])
            started=time.time();child=op.launch(key);identity=b.proc(child.pid)
            state.update(stage=key.upper(),started=started,driver=identity)
            state['runs'].append({'key':key,'identity':identity,'namespace':row['namespace'],'run':row['run']})
            seen={}
            last_proof=None
            while child.poll() is None:
                current=scope_state(op,b,p,row,identity)
                seen.update({(x['pid'],x['start']):x for x in current['tree']})
                state['runs'][-1]['processes']=list(seen.values())
                if current['outside']:
                    b.signal_exact_driver(op,row,identity,'U_GPU_SCOPE_VIOLATION',stage)
                    raise RuntimeError('U GPU scope violation')
                limit=p['smoke_timeout'] if key=='smoke' else p['formal_timeout']
                if stopping or time.time()-started>limit:
                    b.signal_exact_driver(op,row,identity,'U_STOP_OR_TIMEOUT',stage)
                    raise RuntimeError('U stop or timeout')
                free=os.statvfs(row['run']).f_bavail*os.statvfs(row['run']).f_frsize
                if free<40*1024**3:
                    b.signal_exact_driver(op,row,identity,'U_DISK_FLOOR',stage)
                    raise RuntimeError('Output disk below 40 GiB')
                state.update(time=time.time(),scope={'gpu':current['gpu'],'actors':len(current['actors'])})
                if key=='formal' and last_proof is None:
                    m=scalars(row['run'])
                    good={k:v[-1] for k,v in m.items() if v and ('update_step' in k or k=='env/success_once' or 'ugrow' in k)}
                    if ('env/success_once' in good and any('ugrow' in k for k in good)
                            and all(v['time']>=started for v in good.values())):
                        last_proof={'time':time.time(),'identity':identity,'metrics':good,'scope':current['gpu']}
                        save(stage/'formal-first-round.json',last_proof,True)
                save(stage/'status.json',state)
                time.sleep(15)
            rc=child.returncode;op.finished(key,rc)
            for _ in range(20):
                current=scope_state(op,b,p,row,identity)
                if current['released'] and all(not b.same(x) for x in seen.values()):break
                time.sleep(3)
            assert current['released'] and all(not b.same(x) for x in seen.values()),'U resource cleanup incomplete'
            save(stage/(key+'-released.json'),{'time':time.time(),'exit_code':rc,'scope':current},True)
            assert rc==0,(key,rc)
            if key=='smoke':save(stage/'smoke-passed.json',smoke_gate(p,row),True)
        state['stage']='COMPLETE'
    except BaseException:
        state.update(stage='FAILED',error=traceback.format_exc())
    finally:
        if child is not None and child.poll() is None:
            b.signal_exact_driver(op,row,identity,'U_FINAL_CLEANUP',stage)
            for _ in range(40):
                if child.poll() is not None:break
                time.sleep(3)
        state['time']=time.time();save(stage/'status.json',state)
        # Return protocol is supplied by the reviewed lease module. A final
        # receipt never claims release while actor/graphics evidence remains.
        current=scope_state(op,b,p,row,identity) if row else None
        for _ in range(20):
            if not current or current['released']:break
            time.sleep(3)
            current=scope_state(op,b,p,row,identity)
        released=bool(current and current['released'])
        save(stage/'terminal.json',{'time':time.time(),'operation_id':p['operation_id'],'boot_id':p['boot_id'],
             'gpu':p['gpu'],'gpu_uuid':p['gpu_uuid'],'decision':state['stage'],
             'runs':state['runs'],'released':released,'scope':current},True)
        if not released:raise RuntimeError('RLT return waits for proven U release')
        # Filled with exact reviewed lease API before deployment.
        if p.get('return_enabled'):
            for _ in range(120):
                returned=lease.restore(Path(p['lease_dir']),stage/'terminal.json')
                save(stage/'return-status.json',returned)
                if returned['first_round_verified']:break
                if not returned['alive']:raise RuntimeError('Returned RLT driver exited')
                time.sleep(15)
            else:
                save(stage/'return-unverified.json',{'time':time.time(),'status':'NEEDS_ATTENTION','last':returned},True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('command',choices=['driver','owner']);parser.add_argument('key',nargs='?')
    args=parser.parse_args()
    if args.command=='driver':configure(args.stage)[0].driver(args.key)
    else:owner(args.stage)
