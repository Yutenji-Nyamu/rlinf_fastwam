"""Fresh owner: borrow four RLT GPUs, run WM, resume unchanged Dojo, return RLT."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from common import HERE, HELPER, atomic as atomic_json, bind_config, load_base, own_path, read, sha
parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--cycle-dir',type=Path,required=True)
parser.add_argument('--preparation-dir',type=Path,required=True)
args=parser.parse_args()
F=own_path(args.preparation_dir)
ready=read(F/'ready.json')
base,m=load_base(ready['base_source_dir'])
from process_guard import error_context
from hang_watchdog import identity
from wm_stage import run_stage
args.config,cfg,P,R=bind_config(args.config)
STAGE=own_path(ready['cycle_dir'])
assert args.cycle_dir.resolve()==STAGE.resolve()
evaluation_run=R
run=own_path(ready['attempt_dir'],exists=False)
assert run.parent==R and F.parent==R and not run.exists()
scripts=HERE
assert ready['benchmark_unchanged'] is True
assert str(args.config)==ready['config_path'] and sha(args.config)==ready['config_sha256']
assert sha(R/'plan.json')==ready['prior_plan_sha256']
assert sha(F/'wm-spec.json')==ready['wm_spec_sha256']
for file,digest in ready['source_sha256'].items():assert sha(HERE/file)==digest,file
assert sha(STAGE/HELPER)==read(STAGE/'plan.json')['script_sha256']
lock=(R/'pipeline.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (STAGE/'resumed-dispatched.json').exists()
if ready.get('reuse_borrowed_cycle', False):
    assert (STAGE/'rlt-stopped.json').is_file()
    previous = own_path(ready['previous_attempt'])
    assert read(previous/'pipeline-final.json')['rlt_dispatched'] is False
    released = read(previous/'dojo-release.json')
    assert released['all_workers_stopped'] and released['cleanup_receipt']['gpus_released']
    assert not m.gpu_processes([4,5,6,7]), 'Borrowed GPUs have not been released'
else:
    assert not (STAGE/'stop-attempt.json').exists()
run.mkdir(exist_ok=False)
atomic_json(R/'active-continuation.json',{'attempt_dir':str(run),'cycle_dir':str(STAGE),
            'config':str(args.config),'time':time.time(),**identity(os.getpid())})
rlt_python='/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
rlt=[rlt_python,'-u','-B',str(STAGE/HELPER),'--cycle-dir',str(STAGE)]
policy=str(Path(cfg['sim_env'])/'bin/python')
dojo=[policy,'-u','-B',str(base/'dojo_sweep.py'),'--config',str(args.config)]

stop_requested = False
wm_stop_requested = False
in_wm = False
child = None

def signal_stop(signum, frame):
    global stop_requested,wm_stop_requested
    if in_wm:wm_stop_requested=True
    else:stop_requested=True

signal.signal(signal.SIGTERM, signal_stop)
signal.signal(signal.SIGINT, signal_stop)

def state(phase, **values):
    record = dict(time=time.time(), phase=phase, run_id=cfg['run_id'], pid=os.getpid(), continuation=str(run), **values)
    atomic_json(run/'pipeline-current.json', record)
    atomic_json(evaluation_run/'pipeline-current.json', record)
    with (run/'pipeline-events.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)

def command(argv, label, interruptible=False):
    global child
    with (run/(label+'.log')).open('ab') as output:
        child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        pid = child.pid
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        atomic_json(run/(label+'-identity.json'), {'pid':pid,'uid':os.getuid(),'start':int(fields[19])})
        terminated = False
        while child.poll() is None:
            if interruptible and stop_requested and not terminated:
                child.terminate()
                terminated = True
            time.sleep(1)
        rc = child.returncode
        child = None
    atomic_json(run/(label+'-exit.json'), {'time':time.time(),'exit_code':rc,'argv':argv})
    return rc

def managed_identities():
    found = {}
    def add(row):
        if isinstance(row, dict) and row.get('uid') == 20001 and 'pid' in row:
            start = row.get('start', row.get('start_ticks'))
            if start is not None:
                found[(row['pid'],start)] = {'pid':row['pid'],'uid':20001,'start':start}
    p = run/'dojo-controller-identity.json'
    if p.is_file(): add(json.loads(p.read_text()))
    for path in evaluation_run.glob('seed*/worker*/**/process-*.json'):
        add(json.loads(path.read_text()))
    for path in (evaluation_run/'cleanup').glob('*.json'):
        data = json.loads(path.read_text())
        for row in data.get('initial_targets',[]) + data.get('actions',[]): add(row)
    if (run/'wm-release.json').is_file():
        for row in read(run/'wm-release.json')['managed_processes']:add(row)
    return list(found.values())


def verify_rlt_first_round():
    """Observe only; a timeout never stops or relaunches the restored training."""
    started = time.monotonic()
    deadline = started + 1800
    attempt, latest, latest_error = 0, None, None
    while time.monotonic() < deadline and not stop_requested:
        attempted_at = time.monotonic()
        attempt += 1
        status_command = rlt + ['status']
        try:
            response = subprocess.run(status_command, capture_output=True, text=True,
                                      timeout=min(55, max(1, deadline-time.monotonic())))
            (run/f'rlt-status-{attempt:03d}.log').write_text(response.stdout+'\nSTDERR:\n'+response.stderr)
            if response.returncode != 0:
                raise RuntimeError(f'RLT status exit={response.returncode}; see attempt {attempt}')
            parsed = None
            for line in reversed(response.stdout.splitlines()):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and 'all_first_rounds_verified' in value and 'runs' in value:
                    parsed = value
                    break
            assert parsed and parsed['cycle_id'] == args.cycle_dir.name, 'Missing or mismatched RLT status'
            assert set(parsed['runs']) == {'gpu4','gpu5','gpu6','gpu7'}, 'Unexpected restored run set'
            latest, latest_error = parsed, None
            verified = parsed['all_first_rounds_verified'] is True and all(
                row['first_round_verified'] is True for row in parsed['runs'].values())
            record = {'time':time.time(), 'state':'verified' if verified else 'pending',
                      'attempts':attempt, 'elapsed_seconds':time.monotonic()-started,
                      'status':latest, 'error':None, 'training_action':'none; read-only observation'}
            atomic_json(run/'rlt-first-round.json', record)
            if verified:
                state('RLT_FIRST_ROUNDS_VERIFIED', attempts=attempt)
                return record
        except Exception as exc:
            latest_error = repr(exc)
            atomic_json(run/'rlt-first-round.json', {
                'time':time.time(), 'state':'pending', 'attempts':attempt,
                'elapsed_seconds':time.monotonic()-started, 'status':latest,
                'error':latest_error, 'training_action':'none; read-only observation'})
        next_poll = min(deadline, attempted_at+60)
        while time.monotonic() < next_poll and not stop_requested:
            time.sleep(max(0, min(1, next_poll-time.monotonic())))
    pending = {'time':time.time(), 'state':'pending', 'attempts':attempt,
               'elapsed_seconds':time.monotonic()-started, 'status':latest,
               'error':latest_error, 'reason':'observer_stop_requested' if stop_requested else '30_minute_observation_window_ended',
               'training_action':'none; restored training remains running'}
    atomic_json(run/'rlt-first-round.json', pending)
    state('RLT_FIRST_ROUNDS_PENDING', reason=pending['reason'])
    return pending


terminal,error,dojo_code='not_started',None,None
wm_attempted,wm_released=False,False
try:
    state('VERIFYING_SAME_BENCHMARK_AND_RESUME')
    assert command(dojo+['--plan-only'],'final-preflight')==0
    if stop_requested:raise RuntimeError('Stop requested before borrowing GPUs')
    if ready.get('reuse_borrowed_cycle', False):
        state('REUSING_ALREADY_BORROWED_GPUS', cycle=str(STAGE), previous=ready['previous_attempt'])
    else:
        state('STOPPING_CURRENT_FOUR_RLT_RUNS',cycle=str(STAGE))
        assert command(rlt+['stop'],'rlt-stop')==0
    assert (STAGE/'rlt-stopped.json').is_file()
    if stop_requested:raise RuntimeError('Owner stopped before WM stage')
    in_wm,wm_attempted=True,True
    try:
        wm_result=run_stage(read(F/'wm-spec.json'),STAGE,run,m.gpu_processes,state,lambda:wm_stop_requested)
        wm_released=True
    finally:
        in_wm=False
    state('WM_RELEASED_RESUMING_DOJO',wm_release=str(run/'wm-release.json'),
          wm_outcome=wm_result['outcome'],wm_exit_code=wm_result['wm_exit_code'])
    state('EVALUATING',gpus=[4,5,6,7],simultaneous_workers=4,workers_per_gpu=1,
          environments_per_worker=4,expected_episodes=6300,seeds=[0,1,2],wall_time_limit=None)
    dojo_code=command(dojo,'dojo-controller',interruptible=True)
    terminal='completed' if dojo_code==0 else 'failed'
except Exception as exc:
    error=repr(exc)
    state('PIPELINE_ERROR',error=error,error_context=error_context(exc))
finally:
    if (args.cycle_dir/'rlt-stopped.json').exists():
        try:
            assert not wm_attempted or wm_released, 'WM release unverified; refuse Dojo or RLT launch'
            state('CLEANING_THIS_DOJO_SWEEP', dojo_exit_code=dojo_code)
            assert command(dojo+['--cleanup-only'], 'dojo-cleanup') == 0, 'Dojo cleanup failed'
            clean = json.loads((evaluation_run/'cleanup-only-latest.json').read_text())
            assert clean['processes_clear'] and clean['gpus_released']
            release = {'cycle_id':args.cycle_dir.name, 'terminal_status':terminal,
                       'all_workers_stopped':True, 'managed_processes':managed_identities(),
                       'cleanup_receipt':clean, 'time':time.time()}
            atomic_json(run/'dojo-release.json', release)
            if ready.get('restore_rlt_after_dojo', True):
                state('RESTORING_FOUR_RLT_RUNS', terminal_status=terminal)
                assert command(rlt+['resume','--release-receipt',str(run/'dojo-release.json')], 'rlt-resume') == 0, 'RLT restore dispatch failed'
                state('RLT_RESTORE_DISPATCHED', dojo_exit_code=dojo_code)
                verify_rlt_first_round()
            else:
                state('DOJO_FINISHED_RLT_REMAINS_PAUSED', terminal_status=terminal,
                      reason='User priority: WM first, Dojo second; defer RLT until requested')
        except Exception as exc:
            error = (error or '') + '; resource return: ' + repr(exc)
            state('RESOURCE_RETURN_NEEDS_ATTENTION', error=error, error_context=error_context(exc))
    elif (args.cycle_dir/'stop-attempt.json').exists():
        # Signals may have been sent before checkpoint/release validation failed.
        # The RLT cycle intentionally requires inspection of this partial state.
        error = (error or '') + '; RLT stop started but no complete stop receipt; inspect cycle before restore'
        state('RESOURCE_RETURN_NEEDS_ATTENTION', error=error,
              stop_attempt=str(args.cycle_dir/'stop-attempt.json'))
    first_round = run/'rlt-first-round.json'
    final = {'time':time.time(),'terminal_status':terminal,'dojo_exit_code':dojo_code,
             'error':error,'rlt_dispatched':(args.cycle_dir/'resumed-dispatched.json').exists(),
             'wm_attempted':wm_attempted,'wm_released':wm_released,
             'wm_result':read(run/'wm-release.json') if (run/'wm-release.json').exists() else None,
             'rlt_first_round':json.loads(first_round.read_text()) if first_round.is_file() else None}
    atomic_json(run/'pipeline-final.json', final)
    print(json.dumps(final), flush=True)
sys.exit(0 if terminal=='completed' and not error else 1)
