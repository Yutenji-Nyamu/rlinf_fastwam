"""One explicit retry after reconciled OFT learning; reuse the already borrowed GPUs."""
import argparse, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

R = Path('/data/chenyiteng/projects/wan-goal-sz3')
P = Path('/data/chenyiteng/projects/robodojo-openwam-sz3')
D = P/'runs/sz3_pi05_official_6300_n4_dual_20260929_r2'
PREVIOUS = D/'continuation-20260930-wan-goal-v1'
BRIDGE = P/'scripts/wm-bridge-20261001-v2'
CYCLE = P/'rlt-cycle-sz3-wan-goal-20260930-v1'
ATTEMPT = D/'continuation-20261001-wan-goal-v2'
PREP = D/('prepare-'+ATTEMPT.name)
W1 = R/'runs/wan-goal-sz3-20260930-r1'
W2 = R/'runs/wan-goal-sz3-20261001-r2'
RECONCILED = W1/'oft-smoke-control/learning-reconciled.json'
PY = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
sys.path.insert(0, str(R/'scripts/resource_switch'))
from common import account, alive, exclusive, identity, read, sha

def review():
    account()
    manifest=read(R/'scripts/monitor-retry-reviewed.json')
    for relative,digest in manifest['source_sha256'].items():
        assert sha(R/'scripts'/relative)==digest, relative
    return manifest

def released():
    active=read(D/'active-continuation.json')
    assert active['attempt_dir']==str(PREVIOUS) and active['cycle_dir']==str(CYCLE)
    assert not alive(active), 'Previous owner still alive'
    final=read(PREVIOUS/'pipeline-final.json')
    assert final['error'] is None and final['rlt_dispatched'] is False and final['wm_released'] is True
    clean=read(PREVIOUS/'dojo-release.json')
    assert clean['all_workers_stopped'] and clean['cleanup_receipt']['gpus_released']
    assert (CYCLE/'rlt-stopped.json').is_file() and not (CYCLE/'resumed-dispatched.json').exists()

def reconcile():
    report=read(W1/'oft-smoke-control/independent-verification.json')
    assert report['audited_upstream_commit']=='d34d4c320d08cb982de034aa9a011f08dc0fa217'
    assert report['errors']==['Provided process receipt does not confirm normal completion']
    assert report['metrics']['ok'] and report['weight_change']['ok']
    assert sorted(x['step'] for x in report['checkpoints'] if x['ok'])==[1,2]
    assert not report['ok'] and not report['process_exit_confirmed']
    outer=read(W1/'wm-exit.json');inner=read(W1/'oft-smoke/wm-exit.json')
    assert 'Cannot verify registered WM PID' in outer['error'] and '/environ' in outer['error']
    assert inner['exit_code']==-15
    clean=read(W1/'wm-cleanup.json')
    assert clean['processes_clear'] and clean['gpus_released'] and clean['ray']['stopped']
    files=[W1/'oft-smoke-control/independent-verification.json', W1/'wm-exit.json',
           W1/'oft-smoke/wm-exit.json', W1/'wm-cleanup.json', PREVIOUS/'wm-release.json',
           PREVIOUS/'dojo-release.json', PREVIOUS/'pipeline-final.json']
    evidence=dict(time=time.time(),status='LEARNING_VERIFIED_MONITOR_FAILURE_RECONCILED',
        source_commit=report['audited_upstream_commit'], learning_verified=True, resources_released=True,
        original_verifier_ok=False, original_stage_exit_code=-15,
        reason='Two genuine GRPO updates and saved changed weights verified independently; outer monitor interrupted process teardown on registered /proc/environ read failure.',
        learning_evidence={k:report[k] for k in ('metrics','checkpoints','weight_change')},
        resource_policy='WM first, then original Dojo; RLT stays paused.',
        scope='Accept completed OFT learning evidence only; preserve original abnormal exit and do not replay its budget.',
        evidence_sha256={str(f):sha(f) for f in files})
    exclusive(RECONCILED,evidence)

def prepare():
    manifest=review();released()
    assert not BRIDGE.exists() and not PREP.exists() and not ATTEMPT.exists() and not W2.exists()
    with (D/'pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reconcile()
        BRIDGE.mkdir(mode=0o700)
        for relative in manifest['source_sha256']:
            if relative.startswith('resource_switch/'):
                source=R/'scripts'/relative;shutil.copy2(source,BRIDGE/source.name)
    spec=read(D/'prepare-continuation-20260930-wan-goal-v1/wm-spec.json')
    spec['run_dir']=str(W2)
    spec['command'] += ['--completed-oft-evidence',str(RECONCILED)]
    spec['cleanup_command']=[PY,'-u','-B',str(BRIDGE/'cleanup_owned.py')]
    spec['ray'].update(namespace='wan_goal_sz3_20261001_r2_sequence',temp_dir='/data/chenyiteng/wr/wg1001r2s')
    specpath=R/'scripts/wm-spec-20261001-r2.json';exclusive(specpath,spec)
    config=P/'scripts/eval-recovery-20260930-v3/config.json'
    command=[PY,'-u','-B',str(BRIDGE/'prepare_switch.py'),'--config',str(config),
        '--base-source-dir',str(config.parent),'--cycle-dir',str(CYCLE),
        '--continuation-name',ATTEMPT.name,'--wm-spec',str(specpath),
        '--skip-prior-first-round','--defer-rlt-restore','--reuse-borrowed-cycle']
    subprocess.run(command,check=True)
    plan=read(R/'scripts/launch_plan.json')
    plan.update(bridge_source=str(BRIDGE.relative_to(P)),continuation=ATTEMPT.name,
        wm_run=W2.name,wm_spec=str(specpath),completed_oft_evidence=str(RECONCILED),
        reuse_borrowed_cycle=True,restore_rlt_after_dojo=False)
    exclusive(R/'scripts/launch_plan_v2.json',plan)
    print(json.dumps({'ready':str(PREP/'ready.json'),'reconciled':str(RECONCILED),'run':str(W2)}))

def launch():
    review();released()
    ready=read(PREP/'ready.json')
    assert ready['reuse_borrowed_cycle'] and not ready['restore_rlt_after_dojo']
    assert not ATTEMPT.exists() and not W2.exists()
    for file,digest in ready['source_sha256'].items():assert sha(BRIDGE/file)==digest
    argv=[PY,'-u','-B',str(BRIDGE/'continue_pipeline.py'),'--config',ready['config_path'],
          '--cycle-dir',str(CYCLE),'--preparation-dir',str(PREP)]
    exclusive(PREP/'launch-intent.json',{'time':time.time(),'argv':argv,'reviewed_manifest_sha256':sha(R/'scripts/monitor-retry-reviewed.json')})
    with (PREP/'owner.log').open('xb') as stream:
        child=subprocess.Popen(argv,cwd=P,stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
        row=identity(child.pid)
    exclusive(PREP/'launch-identity.json',row)
    print(json.dumps({'owner':row,'command':argv,'log':str(PREP/'owner.log')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    (launch if parser.parse_args().launch else prepare)()
