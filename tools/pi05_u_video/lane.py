"""Finite existing-style per-card inference -> U/video -> RLT queue."""
import json
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

S = Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0, str(S / 'ops'))
from common import read, save, proc, same, exact_signal, gpus

R = Path(__file__).resolve().parents[2]
O = Path(os.environ['U_VIDEO_ROOT'])
D = Path(os.environ['U_VIDEO_TRANSACTION'])
g = sys.argv[1]
manifest = read(O / 'manifest.json')
cards = [str(v) for v in manifest['physical_gpus']]
assert len(cards) == 2 and g in cards and os.getuid() == manifest['plan_before']['uid']
me = proc(os.getpid())
python = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
env = dict(os.environ, **read(O / f'environment-g{g}.json'), U_VIDEO_TOKEN=f'{D.name}-g{g}')
child = None
started_gpu = False
prior_acceptance = Path(os.environ['U_VIDEO_ACCEPTED_TRANSACTION']) if os.environ.get('U_VIDEO_ACCEPTED_TRANSACTION') else None
save(D / f'lane-g{g}-identity.json', me)


def stop(*_):
    if child and child.poll() is None:
        exact_signal(proc(child.pid), signal.SIGTERM)
    raise SystemExit(143)


signal.signal(signal.SIGTERM, stop)


def cleanup():
    owned = []
    for card in gpus().values():
        for row in card['processes']:
            try:
                e = dict(v.split('=', 1) for v in (Path('/proc') / str(row['pid']) / 'environ').read_text().split('\0') if '=' in v)
            except OSError:
                continue
            if e.get('U_VIDEO_TOKEN') == env['U_VIDEO_TOKEN']:
                identity = proc(row['pid'])
                assert identity['uid'] == os.getuid()
                if identity not in owned:
                    owned.append(identity)
    for identity in owned:
        exact_signal(identity, signal.SIGTERM)
    if owned:
        time.sleep(2)
    clear = not gpus()[int(g)]['processes']
    if started_gpu:
        assert clear, 'Assigned card did not clear; do not launch next job'
    return dict(time=time.time(), card_clear=clear, owned_signalled=owned)


def index():
    with (D / 'index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [read(path) for path in sorted(D.glob('*_b*-result.json'))]
        if prior_acceptance:
            rows += [read(path) for path in sorted(prior_acceptance.glob('*_b*-result.json'))]
        save(O / 'summary.json', dict(time=time.time(), validated_batches=sum(r['state'] == 'VALIDATED' for r in rows),
                                      episodes=16 * sum(r['state'] == 'VALIDATED' for r in rows),
                                      failed_batches=sum(r['state'] == 'FAILED' for r in rows), batches=rows))


try:
    plan = read(D / f'lane-g{g}-plan.json')
    predecessor = plan['predecessor']
    save(D / f'lane-g{g}-status.json', dict(time=time.time(), phase='WAIT_PREDECESSOR',
                                        predecessor=predecessor['external_release']))
    while same(predecessor['external_identity']) or not Path(predecessor['external_release']).is_file():
        time.sleep(15)
    assert not gpus()[int(g)]['processes'], 'Predecessor has not cleared its assigned card'
    accepted = False
    if prior_acceptance:
        prior = read(prior_acceptance / f'acceptance-g{g}.json')
        assert prior['validation']['passed'] and prior['validation']['video_passed']
        accepted = True
    failed = []
    for job in plan['jobs']:
        if list(D.glob('lane-g*-error.json')):
            raise RuntimeError('Peer video lane failed validation or cleanup')
        path = Path(job['config'])
        c = read(path)
        out = Path(c['output'])
        assert not out.exists(), 'No implicit retries or overwrites'
        c['acceptance'] = not accepted
        save(path, c)
        assert c['gpu'] == int(g) and not gpus()[int(g)]['processes']
        started_gpu = True
        record = dict(time=time.time(), phase='RUNNING', config=str(path), gpu=int(g))
        with (D / f'{out.name}-g{g}.log').open('x') as log:
            child = subprocess.Popen([python, '-u', '-B', str(R / 'tools/pi05_u_video/run_batch.py'), str(path)],
                                     cwd=R, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            record['child'] = proc(child.pid)
            save(D / f'lane-g{g}-status.json', record)
            deadline = time.monotonic() + 3600
            rc = None
            while rc is None:
                try:
                    rc = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    progress = out / 'progress.json'
                    stalled = progress.exists() and time.time() - progress.stat().st_mtime > 600
                    if stalled or time.monotonic() > deadline:
                        exact_signal(record['child'], signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            exact_signal(record['child'], signal.SIGKILL)
                            child.wait(timeout=20)
                        rc = 124
        clear = cleanup()
        if rc == 78:
            raise RuntimeError('Graphics bootstrap rejected startup; stop instead of skipping tasks')
        if rc != 0:
            if (out / 'attention_error.json').exists():
                raise RuntimeError('Attention method validation failed; do not skip through the dataset')
            failed.append(str(path))
            save(D / f'{out.name}-result.json', dict(time=time.time(), config=str(path), state='FAILED',
                                                    returncode=rc, clearance=clear))
            index()
            continue  # User requested skipping native task errors; no retries.
        save(D / f'lane-g{g}-status.json', dict(record, phase='VALIDATING'))
        with (D / f'{out.name}-validation.log').open('x') as log:
            child = subprocess.Popen([python, '-B', str(R / 'tools/pi05_u_video/validate.py'), str(path)],
                                     env=dict(env, CUDA_VISIBLE_DEVICES=''), cwd=R,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                exact_signal(proc(child.pid), signal.SIGTERM)
                child.wait(timeout=20)
                raise
        assert rc == 0, 'Recording/U validation failed; do not expand collection'
        validation = read(out / 'validation.json')
        assert validation['passed'] and validation['video_passed']
        save(D / f'{out.name}-result.json', dict(time=time.time(), config=str(path), state='VALIDATED',
                                                returncode=0, validation=validation, clearance=clear))
        index()
        if not accepted:
            accepted = True
            save(D / f'acceptance-g{g}.json', dict(time=time.time(), config=str(path), validation=validation))
            save(D / f'lane-g{g}-status.json', dict(time=time.time(), phase='WAIT_PEER_ACCEPTANCE'))
            other = next(card for card in cards if card != g)
            while not (D / f'acceptance-g{other}.json').exists():
                assert not (D / f'lane-g{other}-error.json').exists()
                assert not (D / f'lane-g{other}-released.json').exists(), 'Peer finished without accepted video batch'
                time.sleep(15)
    save(D / f'lane-g{g}-released.json', dict(time=time.time(), identity=me, completed=True,
                                            dataset_complete=not failed, failed_jobs=failed, clearance=cleanup()))
except BaseException:
    save(D / f'lane-g{g}-error.json', dict(time=time.time(), error=traceback.format_exc()))
    clear = cleanup()
    save(D / f'lane-g{g}-released.json', dict(time=time.time(), identity=me, completed=False,
                                            reason='Original owner resumes RLT only after predecessor/lane exit and GPU clearance', clearance=clear))
    raise
