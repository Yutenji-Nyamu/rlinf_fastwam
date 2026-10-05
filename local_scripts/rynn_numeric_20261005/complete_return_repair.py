"""Finish the prepared short-path return; correct only the prelaunch file list.

HOME profile hashes belong to the existing graphics manifest validator, while
the lifecycle's frozen_files validator admits only /data/chenyiteng paths.
"""
import fcntl,hashlib,importlib.util,json,os,socket,sys
from pathlib import Path
D=Path('/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rynn-numeric-v1')
ROOT=D/'rlt-return-repair-v2';NEW=ROOT/'rynn-numeric-v1-gpu4';OLD=D/'prepared/rynn-numeric-v1-gpu4'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    p.chmod(0o600)
def main():
    assert os.getuid()==20001 and socket.gethostname()=='h100-gpu01'
    spec=importlib.util.spec_from_file_location('complete_numeric_return',NEW/'rlt_returned_cycle.py')
    R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R);R.install_helper(NEW);H=R.H
    assert not (ROOT/'repaired.json').exists()
    with (OLD/'operation.lock').open('a') as a,(NEW/'operation.lock').open('a') as b:
        fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in (OLD,NEW):
            assert not any((stage/n).exists() for n in ('gpu4-launch-attempt.json','gpu4-launched.json','resumed-dispatched.json'))
        assert not H.same(read(D/'run/owner-identity.json')) and not H.gpu_processes([4])
        p=read(NEW/'plan.json');row=p['runs']['gpu4'];live=H.actors(p)
        assert not H.same(row['original_identity']) and not H.active(live,row['original_namespace']) and not H.active(live,row['namespace'])
        untouched=read(D/'run/owner-plan.json')['untouched_drivers'];assert all(H.same(x['identity']) for x in untouched.values())
        changed=read(ROOT/'scope/semantic-equivalence.json');assert len(changed)==2
        original_plan_sha=sha(NEW/'plan.json')
        assert original_plan_sha==read(ROOT/'prepared.json')['plan_sha256']
        for path,proof in changed.items():
            assert path.startswith('/home/chenyiteng/.nv/nvidia-application-profiles-rc.d/')
            assert p['frozen_files'][path]==proof['sha256']==sha(Path(path))
            del p['frozen_files'][path]
        p['frozen_files'][str(Path(__file__))]=sha(Path(__file__))
        before=ROOT/'plan-before-path-domain-correction.json';before.write_bytes((NEW/'plan.json').read_bytes())
        new_bytes=(json.dumps(p,indent=2)+'\n').encode();temp=NEW/'plan.corrected.tmp';temp.write_bytes(new_bytes);temp.chmod(0o600);os.replace(temp,NEW/'plan.json')
        save(ROOT/'prelaunch-path-domain-correction.json',dict(before=original_plan_sha,after=sha(NEW/'plan.json'),
            removed_only_profile_keys=list(changed),validation='Profile hashes remain mandatory in the derived graphics manifest and runtime; no profile or card rule changed.'))
        R.load_plan(NEW)
        scope_spec=importlib.util.spec_from_file_location('check_named_scope',ROOT/'scope/bootstrap/graphics_scope_runtime.py')
        scope=importlib.util.module_from_spec(scope_spec);scope_spec.loader.exec_module(scope)
        scope.read_manifest(ROOT/'scope/scope.json')
        answer=H.resume(NEW,D/'run/diagnostic-release.json');assert answer['resumed_dispatched']
        result=dict(time=H.now(),result=answer,lifecycle_path=str(NEW),new_run=row['new_run'],
            original_final_preserved_sha256=sha(D/'run/final.json'),plan_sha256=sha(NEW/'plan.json'),
            return_sha256=sha(NEW/'resumed-dispatched.json'),gpu4_launch=read(NEW/'gpu4-launched.json'),
            untouched_drivers_live={k:H.same(x['identity']) for k,x in untouched.items()})
        save(ROOT/'repaired.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
