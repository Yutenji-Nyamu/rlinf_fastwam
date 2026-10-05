"""Reuse the verified named graphics scope; do not change any user profile."""
import importlib.util,json,os,sys
from pathlib import Path

def install_and_verify(plan):
    M=sys.modules['frozen_multigpu_owner'];scope=plan['graphics_scope']
    assert os.getuid()==20001
    assert (Path(plan['lifecycle_path'])/'rlt-stopped.json').is_file()
    assert not M.H.gpu_processes([4,5,6,7])
    fragment=M.read(scope['environment_fragment_file']);manifest=Path(fragment['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'])
    active=M.read(scope['activation_receipt']);m=M.read(manifest)
    assert active['status']=='active' and active['uid']==20001
    assert active['manifest']==str(manifest) and active['manifest_sha256']==M.sha(manifest)
    assert m['physical_gpus']==[4,5,6,7] and m['cpu_full_mask_target']==4
    runtime=Path(m['runtime_path']);assert M.sha(runtime)==m['runtime_sha256']==plan['source_sha256'][str(runtime)]
    spec=importlib.util.spec_from_file_location('bell_existing_scope',runtime);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.read_manifest(manifest)
    before=active['environment_fragment']
    assert all(fragment[k]==before[k] for k in before if k!='PYTHONPATH')
    assert fragment['PYTHONPATH'].split(':')[0]==before['PYTHONPATH'].split(':')[0]
    assert plan['repo'] in fragment['PYTHONPATH'].split(':')
    return dict(time=M.H.now(),status='scope_activated',native_probe_verified=False,
        base_environment_sha256=M.sha(plan['environment_file']),
        environment_fragment_sha256=M.sha(scope['environment_fragment_file']),
        scope_manifest_path=str(manifest),scope_manifest_sha256=M.sha(manifest),
        activation_receipt=scope['activation_receipt'],activation_receipt_sha256=M.sha(scope['activation_receipt']),
        reused_existing_scope=True,profile_changes=False)
