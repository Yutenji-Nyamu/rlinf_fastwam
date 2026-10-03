"""Resume one prepared failed-return repair and verify its first real round."""
import argparse,fcntl,hashlib,importlib.util,json,os,socket,time,traceback
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args();stage=args.stage
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    root=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003').resolve()
    assert stage.resolve().is_relative_to(root) and stage.name=='gpu4-return-repair-v1'
    source=stage/'opendw_smoke_gpu4_cycle.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='7c019fea53ef7b27790bdbacb8b072cb31c0f2a55b89cc496fa56065582c0761'
    spec=importlib.util.spec_from_file_location('repair_cycle',source);C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C);C.install_helper(stage);H=C.H;plan=C.load_plan(stage)
    lock=(stage/'repair-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (stage/'repair-owner-identity.json').exists() and not (stage/'resumed-dispatched.json').exists()
    H.save(stage/'repair-owner-identity.json',dict(H.proc(os.getpid()),time=H.now()))
    error=None;result=None
    try:
        C.protected_check(plan)
        with (stage/'operation.lock').open('a') as operation:
            fcntl.flock(operation,fcntl.LOCK_EX|fcntl.LOCK_NB)
            H.resume(stage,stage/'repair-release.json')
        deadline=time.monotonic()+1200
        while True:
            result=H.status(stage);H.atomic(stage/'repair-status.json',result)
            if result['all_first_rounds_verified']:
                H.save(stage/'repair-first-round.json',result);break
            if result['runs']['gpu4'].get('finished'):
                raise RuntimeError('Repaired RLT exited before first round')
            if time.monotonic()>deadline:
                raise TimeoutError('Repair dispatched; first round still pending, healthy process is preserved')
            time.sleep(15)
    except BaseException as exc:
        error={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()}
    H.save(stage/'repair-final.json',{'time':H.now(),'status':'verified' if error is None else 'attention','error':error,'first_round_verified':(stage/'repair-first-round.json').exists(),'last_status':result})
    if error:raise SystemExit(1)

if __name__=='__main__':main()
