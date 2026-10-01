"""Single-use monitor repair retry; unchanged formal configuration and direct RLT return."""
import argparse,fcntl,json,shutil,socket,subprocess,sys,time
from pathlib import Path

R=Path('/data/chenyiteng/projects/wan-goal-sz3')
P=Path('/data/chenyiteng/projects/robodojo-openwam-sz3')
D=P/'runs/sz3_pi05_official_6300_n4_dual_20260929_r2'
PREVIOUS=D/'continuation-20261001-wan-goal-v5'
PREVIOUS_CYCLE=P/'rlt-cycle-sz3-wan-goal-20261001-wake-v3'
BRIDGE=P/'scripts/wm-bridge-20261001-v6'
CYCLE=P/'rlt-cycle-sz3-wan-goal-20261001-repair-v1'
ATTEMPT=D/'continuation-20261001-wan-goal-v6'
PREP=D/('prepare-'+ATTEMPT.name)
WM=R/'runs/wan-goal-sz3-20261001-r6'
PY='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
MANIFEST=R/'scripts/monitor-repair-reviewed-20261001.json'
sys.path.insert(0,str(R/'scripts/resource_switch'))
from common import account,alive,exclusive,identity,read,sha
from prepare_switch import verify_prior_return

def review():
    account();manifest=read(MANIFEST)
    for relative,digest in manifest['source_sha256'].items():
        assert sha(R/'scripts'/relative)==digest,relative
    for absolute,digest in manifest['config_sha256'].items():
        assert sha(absolute)==digest,absolute
    assert sha(R/'logs/tmpfs-compile-check-20261001.json')==manifest['cpu_compile_receipt_sha256']
    checked=read(R/'logs/tmpfs-compile-check-20261001.json')
    assert checked['ok'] and all(x['forward_backward_match'] and not x['cuda_initialized'] for x in checked['children'])
    oft=read(manifest['completed_oft_evidence'])
    assert oft['status']=='LEARNING_VERIFIED_MONITOR_FAILURE_RECONCILED'
    assert oft['source_commit']=='d34d4c320d08cb982de034aa9a011f08dc0fa217'
    assert oft['learning_verified'] and oft['resources_released']
    for file,digest in oft['evidence_sha256'].items():assert sha(file)==digest
    pi05=read(manifest['completed_pi05_evidence'])
    assert sha(manifest['completed_pi05_evidence'])==manifest['completed_pi05_evidence_sha256']
    assert pi05['status']=='ONE_VALID_PI05_GRPO_UPDATE_VERIFIED'
    assert pi05['learning_verified'] and pi05['resources_released'] and pi05['process_exit_confirmed']
    assert pi05['completed_runner_epochs']==2 and pi05['effective_update_count']==1
    assert pi05['formal_runner_epochs']==1000 and pi05['formal_initialization']=='original_fixed_SFT'
    assert pi05['no_additional_smoke_budget'] and all(x['all_finite'] for x in pi05['checkpoint_floating_parameters'])
    for file,digest in {**pi05['evidence_sha256'],**pi05['approved_model_source_sha256']}.items():assert sha(file)==digest
    checks=read(manifest['repair_cpu_receipt'])
    assert sha(manifest['repair_cpu_receipt'])==manifest['repair_cpu_receipt_sha256']
    assert checks['ok'] and checks['cuda_initialized'] is False
    assert all(x['exit_code']==0 for x in checks['tests'])
    for relative,digest in checks['source_sha256'].items():assert sha(R/'scripts'/relative)==digest,relative
    return manifest

def prior_returned():
    active=read(D/'active-continuation.json')
    assert active['attempt_dir']==str(PREVIOUS) and active['cycle_dir']==str(PREVIOUS_CYCLE)
    assert not alive(active),'Previous owner remains alive'
    assert verify_prior_return(PREVIOUS,PREVIOUS_CYCLE,False)=='pipeline-final.json'

def prepare():
    manifest=review();prior_returned()
    for target in (BRIDGE,CYCLE,ATTEMPT,PREP,WM):assert not target.exists(),str(target)
    for port in (63842,63844,63845):
        with socket.socket() as s:
            s.settimeout(1);assert s.connect_ex(('127.0.0.1',port))!=0,port
    with (D/'pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        BRIDGE.mkdir(mode=0o700)
        for relative in manifest['source_sha256']:
            if relative.startswith('resource_switch/'):
                source=R/'scripts'/relative;shutil.copy2(source,BRIDGE/source.name)
    spec=read(D/'prepare-continuation-20261001-wan-goal-v5/wm-spec.json')
    spec['run_dir']=str(WM)
    assert spec['command'][-4:]==['--completed-oft-evidence',manifest['completed_oft_evidence'], '--completed-pi05-evidence',manifest['completed_pi05_evidence']]

    spec['cleanup_command']=[PY,'-u','-B',str(BRIDGE/'cleanup_owned.py')]
    spec['ray'].update(namespace='wan_goal_sz3_20261001_r6_sequence',temp_dir='/data/chenyiteng/wr/wg1001r6s')
    specpath=R/'scripts/wm-spec-20261001-r6.json';exclusive(specpath,spec)
    config=P/'scripts/eval-recovery-20260930-v3/config.json'
    subprocess.run([PY,'-u','-B',str(BRIDGE/'prepare_switch.py'),'--config',str(config),
        '--base-source-dir',str(config.parent),'--cycle-dir',str(CYCLE),
        '--continuation-name',ATTEMPT.name,'--wm-spec',str(specpath),'--skip-prior-first-round','--return-rlt-direct'],check=True)
    ready=read(PREP/'ready.json')
    assert ready['restore_rlt_after_dojo'] and ready['return_rlt_direct'] and not ready['reuse_borrowed_cycle']
    exclusive(R/'scripts/launch_plan_v6.json',dict(time=time.time(),previous_attempt=str(PREVIOUS),
        bridge_source=str(BRIDGE),cycle=str(CYCLE),continuation=str(ATTEMPT),wm_run=str(WM),
        preparation=str(PREP),physical_gpus=[4,5,6,7],restore_rlt_after_dojo=True,return_rlt_direct=True,
        config_unchanged=True,formal_runner_epochs=1000,smoke_replayed=False,
        completed_pi05_evidence=manifest['completed_pi05_evidence'],reviewed_manifest_sha256=sha(MANIFEST)))
    print(json.dumps({'ready':str(PREP/'ready.json'),'wm_run':str(WM),'cycle':str(CYCLE)}))

def launch():
    review();prior_returned();ready=read(PREP/'ready.json')
    assert ready['restore_rlt_after_dojo'] and ready['return_rlt_direct'] and not ready['reuse_borrowed_cycle']
    assert not ATTEMPT.exists() and not WM.exists()
    for file,digest in ready['source_sha256'].items():assert sha(BRIDGE/file)==digest
    argv=[PY,'-u','-B',str(BRIDGE/'continue_pipeline.py'),'--config',ready['config_path'],
          '--cycle-dir',str(CYCLE),'--preparation-dir',str(PREP)]
    exclusive(PREP/'launch-intent.json',{'time':time.time(),'argv':argv,'reviewed_manifest_sha256':sha(MANIFEST)})
    with (PREP/'owner.log').open('xb') as stream:
        child=subprocess.Popen(argv,cwd=P,stdout=stream,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
        row=identity(child.pid)
    exclusive(PREP/'launch-identity.json',row)
    print(json.dumps({'owner':row,'argv':argv,'log':str(PREP/'owner.log')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    (launch if parser.parse_args().launch else prepare)()
