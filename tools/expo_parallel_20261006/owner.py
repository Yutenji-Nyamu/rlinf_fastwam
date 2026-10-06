"""Adopt the exact 60k owner, test 1 then 2 cards, retain the existing RLT return."""
import fcntl
import os
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from common import *
sys.path.insert(0, str(SOURCE / 'tools'))
from expo_smoke_owner import Roster, identity, owned, same, gpu_map
from expo_process import pidfd_open, pidfd_send, pidfd_probe

def gpu_rows():
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=25))
    return [dict(index=i, pid=int(p.findtext('pid')), type=p.findtext('type'), memory=p.findtext('used_memory'))
            for i,g in enumerate(root.findall('gpu')) for p in g.findall('processes/process_info')]

def exact_signal(row, sig):
    assert owned(row)
    fd = pidfd_open(row['pid'])
    try:
        assert owned(row); pidfd_send(fd, sig)
    finally: os.close(fd)

def wait_stopped(row):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        assert owned(row)
        if identity(row['pid'])['state'] in ('T', 't'): return
        time.sleep(.05)
    raise TimeoutError('Exact owner did not stop')

def wait_free(managed):
    end = time.monotonic() + 120; clear = None
    while time.monotonic() < end:
        rows = gpu_rows(); pids = {x['pid'] for x in managed}
        if not any(r['index'] in (4,5,6,7) or r['pid'] in pids for r in rows) and not any(owned(r) for r in managed):
            if clear is None: clear = time.monotonic()
            elif time.monotonic() - clear > 2: return
        else: clear = None
        time.sleep(1)
    raise RuntimeError('Compute/graphics release incomplete')

def environment(devices, phase, source=SOURCE):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(UUIDS[:devices]),
        PYTHONPATH=str(STAGE / 'bootstrap') + ':' + str(source) + ':/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
        LD_LIBRARY_PATH='/home/chenyiteng/tools/cuda-12.9/lib64', VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
        PYTHONFAULTHANDLER='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', TMPDIR=str(STAGE / 'tmp'),
        EXPO_SMOKE_OWNER_SCOPE='expo-parallel-20261006', EXPO_SMOKE_OWNER_PHASE=phase,
        RLINF_EXPO_GPU_SCOPE_MANIFEST=str(STAGE / f'scope-{devices}.json'), __GL_APPLICATION_PROFILE='1')
    env.pop('DISPLAY', None)
    return env

def main():
    assert os.getuid() == 20001 and socket.gethostname() == 'h100-gpu02'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    resource.setrlimit(resource.RLIMIT_CORE, (0,0)); pidfd_probe()
    contract = read(STAGE / 'staged.json'); old = contract['current']; me = identity(os.getpid())
    lock = (STAGE / 'owner.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (STAGE / 'owner.json').exists()
    atomic(STAGE / 'owner.json', dict(owner=me, time=time.time(), previous=old['owner']))
    retired = held = False; active = child = None; managed = []; locks = [lock]; error = None
    stopped = passed = returned = False; requested = [False]
    meta = dict(owner=me, scope='expo-parallel-20261006', control=str(STAGE), train=str(TRAIN),
                cycle=str(CYCLE), max_physical_actions=60000, physical_gpus=[4,5,6,7])
    def state(status, **more):
        meta.update(status=status, time=time.time(), **more); atomic(STAGE / 'current.json', meta)
        if retired: atomic(OLD / 'current.json', meta)
    def pins():
        for p,h in contract['files'].items(): assert sha(p) == h,p
    def no_return():
        assert not any((OLD / name).exists() for name in ('release.json','final.json','rlt-resume-intent.json'))
        assert not (CYCLE / 'resumed-dispatched.json').exists()
    def signalled(_s,_f): requested[0] = True
    for s in (signal.SIGTERM, signal.SIGINT): signal.signal(s, signalled)
    def cleanup(grace=0):
        nonlocal active, child
        if active is None: return
        if grace and owned(active.root):
            active.signal(active.root, signal.SIGTERM); end = time.monotonic() + grace
            while owned(active.root) and time.monotonic() < end: active.scan(); time.sleep(.5)
        result = active.cleanup(); managed.extend(active.rows.values())
        if child is not None: child.wait(timeout=10)
        atomic(STAGE / ('cleanup-' + active.phase + '.json'), result)
        wait_free(managed); active = child = None
    def start(argv, phase, devices, source=SOURCE):
        nonlocal active, child
        pins(); wait_free(managed)
        with (STAGE / (phase + '.log')).open('xb') as f:
            child = subprocess.Popen(argv, cwd=source, env=environment(devices, phase, source), stdin=subprocess.DEVNULL,
                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        anchor = identity(child.pid); active = Roster(anchor, meta['scope'], phase, STAGE / (phase + '-roster.json')); active.write()
        state(phase, child=anchor, physical_gpus=list(range(4,4+devices)), argv=argv)
    def monitor(devices, heartbeat=None, timeout=None):
        begin = time.monotonic(); sample = 0
        while child.poll() is None:
            if requested[0]: raise RuntimeError('Owner stop requested')
            active.scan()
            if timeout and time.monotonic() - begin > timeout: raise TimeoutError('Capacity trial time limit exceeded')
            if heartbeat and heartbeat.exists() and time.time() - heartbeat.stat().st_mtime > 900:
                raise TimeoutError('EXPO heartbeat stale >900s')
            if time.monotonic() - sample > 10:
                pids = {r['pid'] for r in active.rows.values()}; rows = [r for r in gpu_rows() if r['pid'] in pids]
                assert not any(r['index'] not in range(4,4+devices) for r in rows), 'Compute/graphics escaped selected GPUs'
                with (STAGE / 'gpu-samples.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(), phase=active.phase, processes=rows))+'\n')
                state(meta['status'], gpu_processes=rows); sample = time.monotonic()
            time.sleep(2)
        return child.returncode
    try:
        pins(); no_return(); assert [gpu_map()[i] for i in (4,5,6,7)] == UUIDS
        current = read(OLD / 'current.json')
        assert same(current['owner'],old['owner']) and same(current['child'],old['child'])
        assert current['status'] == 'EXPO_RUNNING' and owned(old['owner']) and owned(old['child'])
        assert sha(TRAIN / 'inputs.json') == contract['inputs_sha256']
        proof = read(CYCLE / 'rlt-stopped.json')
        assert proof['all_original_drivers_stopped'] and proof['all_original_namespaces_empty']
        old_roster = read(OLD / 'expo-roster.json'); assert same(old_roster['root'], old['child'])
        atomic(STAGE / 'handoff-intent.json', dict(time=time.time(), old=old, new_owner=me))
        exact_signal(old['owner'], signal.SIGSTOP); held = True; wait_stopped(old['owner']); no_return()
        active = Roster(old['child'], old['scope'], 'driver60k', STAGE / 'adopted-roster.json')
        for row in old_roster['registered']: active.rows[(row['pid'],row['start_ticks'])] = row
        active.scan(); active.write()
        exact_signal(old['owner'],signal.SIGKILL); retired = True; held = False
        end = time.monotonic()+10
        while owned(old['owner']) and time.monotonic()<end: time.sleep(.1)
        assert not owned(old['owner'])
        for path in [OLD/'owner.lock', TRAIN/'resource-owner.lock', ROOT/'eval10-continuation-20261003/resource-owner.lock',
                     *[q/'owner.lock' for q in QUEUES]]:
            handle=path.open('a+'); fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(handle)
        state('WAITING_SAFE_STOP',child=old['child'])
        atomic(TRAIN/'active-continuation.json',dict(control=str(STAGE),current=str(STAGE/'current.json'),owner=me,cycle=str(CYCLE)))
        for q in QUEUES:
            atomic(q/'active-continuation.json',dict(control=str(STAGE),current=str(STAGE/'current.json'),owner=me,cycle=str(CYCLE)))
            atomic(q/'queue-status.json',dict(time=time.time(),stage1='COMPLETE',roles={'clean':'WAITING_EXPO_60K','combo':'WAITING_EXPO_60K'},control=str(STAGE),owner=me))
        end=time.monotonic()+7200; stop_time=None
        while owned(old['child']):
            assert not requested[0]; active.scan()
            if read(TRAIN/'run/status.json')['phase']=='learner_started':
                exact_signal(old['child'],signal.SIGSTOP); wait_stopped(old['child'])
                if read(TRAIN/'run/status.json')['phase']=='learner_started':
                    stop_time=time.time(); exact_signal(old['child'],signal.SIGTERM); exact_signal(old['child'],signal.SIGCONT); break
                exact_signal(old['child'],signal.SIGCONT)
            assert time.monotonic()<end,'No safe learner boundary'; time.sleep(.2)
        cleanup(grace=180)
        stop=read(TRAIN/'run/stopped.json'); cp=read(TRAIN/'run/checkpoint.json')
        assert stop_time and stop['time']>=stop_time and cp['finite'] and stop['cadence']==cp['cadence']
        stopped=True; state('BACKING_UP_CHECKPOINTS',child=None)
        backup=STAGE/'baseline'; backup.mkdir()
        for name in ('checkpoint-latest.pt','checkpoint-last1.pt','checkpoint.json','resolved.json','stopped.json'):
            shutil.copy2(TRAIN/'run'/name,backup/name)
        shutil.copy2(TRAIN/'inputs.json',backup/'inputs.json')
        atomic(STAGE/'paused.json',dict(time=time.time(),checkpoint=cp,stopped=stop,owner=me,
               replay_index_sha256=sha(TRAIN/'replay/index.json'),managed_processes=managed))
        selected=4
        for devices in (1,2):
            (STAGE/f'trial-{devices}').mkdir()
            start([EXPO_PY,'-u','-B',str(TOOLS/'probe.py'),'--devices',str(devices)],f'trial-{devices}',devices)
            try:
                code=monitor(devices,STAGE/f'trial-{devices}/heartbeat',timeout=2700)
            except TimeoutError as exc:
                code=124
                atomic(STAGE/f'trial-{devices}/timeout.json',dict(time=time.time(),error=str(exc)))
            cleanup()
            receipt=STAGE/f'trial-{devices}/result.json'
            if code==0 and receipt.is_file() and read(receipt)['ok']:
                selected=devices; break
            atomic(STAGE/f'trial-{devices}-failed.json',dict(time=time.time(),returncode=code,receipt=str(receipt)))
        source=OLD/'source'
        if selected!=4:
            state('MIGRATING_DEVICE_METADATA',selected_devices=selected)
            with (STAGE/'migration.log').open('xb') as f:
                subprocess.run([EXPO_PY,'-u','-B',str(TOOLS/'migrate.py'),'--devices',str(selected)],cwd=TOOLS,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(SOURCE),PYTHONDONTWRITEBYTECODE='1'),
                    stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
            source=SOURCE
        atomic(STAGE/'selection.json',dict(time=time.time(),devices=selected,unchanged_method=True,
               reason='isolated_complete_call_passed' if selected!=4 else 'both_trials_failed_restore_original'))
        state('RESUMING_EXPO',selected_devices=selected)
        (TRAIN/'driver-heartbeat').touch()
        argv=[EXPO_PY,'-X','faulthandler','-u','-B',str(source/'examples/embodiment/train_expo_formal.py'),
              '--inputs',str(TRAIN/'inputs.json'),'--run',str(TRAIN/'run'),'--max-physical-actions','60000',
              '--enable-evaluation','--resume',str(TRAIN/'run/checkpoint-latest.pt')]
        start(argv,'formal',selected,source); state('EXPO_RUNNING',selected_devices=selected)
        atomic(TRAIN/'active-continuation.json',dict(control=str(STAGE),current=str(STAGE/'current.json'),owner=me,
               child=active.root,cycle=str(CYCLE),source=str(source),budget=60000,physical_gpus=list(range(4,4+selected))))
        code=monitor(selected,TRAIN/'driver-heartbeat'); cleanup()
        done=read(TRAIN/'run/complete.json') if (TRAIN/'run/complete.json').exists() else {}
        passed=code==0 and done.get('ok') and done.get('budget_completed') and done['cadence']['counters']['physical_actions']==60000
        if not passed: raise RuntimeError('Formal continuation exited '+str(code))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        atomic(STAGE/'error.json',error)
    finally:
        if held and not retired and owned(old['owner']): exact_signal(old['owner'],signal.SIGCONT)
        if retired:
            try:
                if active is not None and owned(active.root) and identity(active.root['pid'])['state'] in ('T','t'):
                    active.signal(active.root,signal.SIGCONT)
                cleanup(grace=180); wait_free(managed); no_return()
                atomic(OLD/'release.json',dict(cycle_id=CYCLE.name,terminal_status='completed' if passed else 'failed',
                    all_workers_stopped=True,managed_processes=managed,physical_gpus=[4,5,6,7],continuation_owner=me))
                state('RESTORING_RLT',error=error)
                argv=[RLT_PY,'-u','-B',str(OLD/'source/tools/expo_extend_20261005/resources.py')]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
                atomic(OLD/'rlt-resume-intent.json',dict(time=time.time(),owner=me,control=str(STAGE)))
                with (STAGE/'rlt-resume.log').open('xb') as f:
                    subprocess.run(argv+['resume'],cwd=OLD/'source/tools/expo_extend_20261005',env=env,
                                   stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
                end=time.monotonic()+1800
                while time.monotonic()<end:
                    with (STAGE/'rlt-status.log').open('w') as f:
                        subprocess.run(argv+['status'],cwd=OLD/'source/tools/expo_extend_20261005',env=env,
                                       stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
                    if read(OLD/'rlt-status.json')['all_first_rounds_verified']: returned=True; break
                    time.sleep(20)
                assert returned,'RLT first resumed round still unverified'
                state('RLT_RESTORED',expo_completed=passed)
                for q in QUEUES:
                    atomic(q/'queue-status.json',dict(time=time.time(),stage1='COMPLETE',roles={'clean':'TRAINING','combo':'TRAINING'},control=str(STAGE)))
            except BaseException:
                error=dict(type='CLEANUP_OR_RETURN_NEEDS_ACTION',previous=error,traceback=traceback.format_exc())
                state('ACTION_REQUIRED',error=error)
            final=dict(time=time.time(),owner=me,expo_completed=bool(passed),rlt_first_rounds_verified=returned,error=error)
            atomic(STAGE/'final.json',final); atomic(OLD/'final.json',final)
    return 0 if passed and returned else 1

if __name__=='__main__': raise SystemExit(main())
