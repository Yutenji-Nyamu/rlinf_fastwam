"""Resume exactly three failed SZ3 jobs using the established driver lifecycle."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ST = None
ops = None


def read(p): return json.loads(Path(p).read_text())


def checked():
    plan = read(ST / 'plan.json')
    assert os.getuid() == plan['uid'] == 20001
    assert socket.gethostname() == 'h100-gpu01'
    assert subprocess.check_output(['git','-C',plan['repo'],'rev-parse','HEAD'],text=True).strip() == plan['head']
    assert not subprocess.check_output(['git','-C',plan['repo'],'status','--porcelain'],text=True).strip()
    assert {key: row['gpus'] for key,row in plan['runs'].items()} == {'gpu5':[5], 'gpu6':[6], 'gpu7':[7]}
    return plan


def atomic(path, value):
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with tmp.open('x') as f: json.dump(value, f, indent=2)
    os.replace(tmp, path)


def owner():
    plan = checked()
    lock = (ST / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ST / 'owner-identity.json').exists(), 'Do not replay'
    ops.save(ST / 'owner-identity.json', {**ops.proc(os.getpid()), 'time':ops.now()})
    for row in plan['runs'].values():
        assert not ops.same(row['original_identity'])
        assert read(Path(row['original_run']) / 'runtime/finished.json')['exit_code'] == 255
        assert not ops.active_actors(plan,row['original_namespace'])
        assert not ops.gpu_pids(row['gpus'])
    allow = Path('/data/chenyiteng/security/ray-guard/training_allowlist.json')
    with (allow.parent/'recovery-allowlist.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        before=read(allow);after=dict(before)
        after['namespaces']=sorted(set(before['namespaces']) | {r['namespace'] for r in plan['runs'].values()})
        ops.save(ST/'allowlist-before.json',before);atomic(allow,after)
    children={}
    try:
        for key in plan['runs']: children[key]=ops.launch(key)
        ops.save(ST/'dispatched.json',{'time':ops.now(),'drivers':{k:ops.proc(c.pid) for k,c in children.items()}})
        watch=Path('/data/chenyiteng/deployment-20260927/rlt-six-task-watch/plan.json')
        with (watch.parent/'route.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            before=read(watch);after=json.loads(json.dumps(before));found=set()
            for target in after['runs'].values():
                for key,row in plan['runs'].items():
                    if target['run']==row['original_run']:
                        assert target['gpus']==row['gpus']
                        target.update(run=row['run'],namespace=row['namespace']);found.add(key)
            assert found==set(plan['runs']), 'Watch route changed; inspect'
            ops.save(ST/'watch-before.json',before);atomic(watch,after)
        states={key:'RUNNING' for key in children}
        while children:
            for key, child in list(children.items()):
                rc=child.poll()
                if rc is not None:
                    ops.finished(key,rc);states[key]='COMPLETED' if rc==0 else 'FAILED'
                    del children[key]
            atomic(ST/'state.json',{'time':ops.now(),'roles':states})
            if children:time.sleep(10)
        ops.save(ST/'finished.json',{'time':ops.now(),'roles':states})
    except BaseException as error:
        # Preserve live training on an owner bookkeeping error. Exact launches are receipts.
        ops.save(ST/'owner-error.json',{'time':ops.now(),'type':type(error).__name__,'error':str(error),'launched':{k:ops.proc(c.pid) for k,c in children.items()}})
        raise


def main():
    global ST,ops
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True)
    p.add_argument('action',choices=['owner','driver']);p.add_argument('key',nargs='?')
    a=p.parse_args();ST=Path(a.stage)
    plan=read(ST/'plan.json')
    spec=importlib.util.spec_from_file_location('established_ops',Path(plan['repo'])/'tools/next_six_20261002/ops.py')
    ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
    ops.ST=ST;ops.checked=checked
    if a.action=='driver':ops.driver(a.key)
    else:owner()


if __name__=='__main__':main()
