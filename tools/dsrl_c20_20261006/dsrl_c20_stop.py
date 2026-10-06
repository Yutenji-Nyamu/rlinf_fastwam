"""One-use C10 stop: retire only its CPU owner, then exact drivers with own cleanup."""
import fcntl,json,os,signal,sys,time
from pathlib import Path
OLD=Path('/data/chenyiteng/deployment-20261006/dsrl-u-lease-v1')
NEW=OLD.with_name('dsrl-u-c20-lease-v2')
CODE=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-pi05-u-sz1-20261006/tools/dsrl_u_20261006/ops')
sys.path.insert(0,str(CODE))
from lease_common import checked,read,save,gpu_released,sha
p,b,op=checked(NEW)
capture=read(NEW/'c10-capture.json')
cpu=read(NEW/'c20-cpu-check.json')
assert set(cpu['roles'])=={'clean','u'}
assert sha(OLD/'plan.json')==capture['old_plan_sha256']
assert not (NEW/'c10-stop-attempt.json').exists()
assert not (OLD/'owner-stopped.json').exists()
state=read(OLD/'status.json')
assert time.time()-state['time']<60 and b.same(capture['owner'])

def protect():
    for ident in capture['protected'].values():
        assert b.same(ident),'Protected process changed: '+str(ident)

protect()
for gpu,row in capture['targets'].items():
    assert state['slots'][gpu]['state']=='FORMAL'
    assert state['slots'][gpu]['namespace']==row['namespace'] and b.same(row['driver'])
    assert not (OLD/'receipts'/('gpu'+gpu+'-restore-attempt.json')).exists()
    actors=op.active_actors(p,row['namespace']);op.validate_scoped_actors(p,actors)
    roots={row['driver']['pid']}|{a['pid'] for a in actors if a.get('pid')}
    row['process_tree']=op.process_tree(roots,p['uid'])
save(NEW/'c10-stop-attempt.json',{'time':time.time(),'capture':capture,'cpu_check_sha256':sha(NEW/'c20-cpu-check.json'),
    'signals':'SIGTERM exact old lease owner, then exact Clean/U driver; no other processes'},True)

def term(ident):
    assert b.same(ident)
    fd=b.pidfd_open(ident['pid'])
    try:
        assert b.same(ident)
        b.pidfd_signal(fd,signal.SIGTERM)
    finally:
        os.close(fd)

term(capture['owner'])
deadline=time.monotonic()+40
while b.same(capture['owner']):
    assert time.monotonic()<deadline,'CPU owner did not exit; drivers retained'
    time.sleep(1)
stopped=read(OLD/'owner-stopped.json')
assert stopped['stopped'] is True and stopped['owner']['pid']==capture['owner']['pid']
assert all(r['state']=='FORMAL' for r in stopped['slots'].values())
locks=[]
for gpu in ('6','7'):
    f=(NEW.parent/('dsrl-rlt-gpu'+gpu+'-lease.lock')).open('a')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
protect()
for gpu in ('6','7'):
    assert not (OLD/'receipts'/('gpu'+gpu+'-restore-attempt.json')).exists()
    term(capture['targets'][gpu]['driver'])
save(NEW/'c10-drivers-signalled.json',{'time':time.time(),'owner_stopped':stopped,'targets':capture['targets']},True)
deadline=time.monotonic()+240
while True:
    protect()
    detail={}
    for gpu,row in capture['targets'].items():
        released,snapshot,actors=gpu_released(p,b,op,gpu,row['namespace'])
        live=[i for i in row['process_tree'].values() if b.same(i)]
        detail[gpu]={'released':released and not live and not b.same(row['driver']),
                    'gpu':snapshot,'actors':actors,'live_recorded':live,'driver_alive':b.same(row['driver'])}
    result={'time':time.time(),'old_owner_alive':b.same(capture['owner']),'targets':detail,
            'protected':capture['protected'],'all_released':all(r['released'] for r in detail.values())}
    save(NEW/'c10-release-status.json',result)
    if result['all_released']:
        assert not result['old_owner_alive']
        save(NEW/'c10-released.json',result,True)
        print(json.dumps(result));break
    assert time.monotonic()<deadline,'Release incomplete; inspect receipt, no kill escalation'
    time.sleep(3)
