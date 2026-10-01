"""Server-only CPU fixtures for the real new owner's WM -> RLT control flow.

Runs continue_pipeline.py through runpy in fresh child interpreters. RLT, Dojo,
and WM releases are explicitly marked fixtures; no Ray, CUDA, SSH, or workload
process is inspected, stopped, resumed, or started. Every writable test path is
inside a new account-owned temporary directory, which is removed on completion.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch


ACCOUNT_ROOT = Path('/data/chenyiteng')
ACCOUNT_UID = 20001
EXPECTED_HOST = 'h100-gpu01'
FIXTURE_SCOPE = 'CPU control-flow fixture only; no real WM, GPU, Ray, Dojo or RLT workload'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def account():
    if sys.platform != 'linux' or os.getuid() != ACCOUNT_UID or socket.gethostname() != EXPECTED_HOST:
        raise RuntimeError('This check requires the authorized SZ3 Linux account')
    if ACCOUNT_ROOT.resolve().stat().st_uid != ACCOUNT_UID:
        raise RuntimeError('Account data root is not owned by the authorized account')


RLT_FIXTURE = r'''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--cycle-dir',type=Path,required=True)
p.add_argument('action',choices=['stop','resume','status'])
p.add_argument('--release-receipt',type=Path)
a=p.parse_args()
cycle=a.cycle_dir
meta=json.loads((cycle/'fixture.json').read_text())
root=Path(meta['root'])
assert cycle.resolve().is_relative_to(root.resolve()) and meta['fixture'] is True
def read(path):return json.loads(path.read_text())
def write(path,value):
    with path.open('x') as f:json.dump(value,f);f.write('\n')
def event(kind):
    with (root/'fixture-events.jsonl').open('a') as f:
        f.write(json.dumps({'fixture':True,'event':kind,'time':time.time()})+'\n')
if a.action=='stop':
    event('RLT_STOP')
    write(cycle/'stop-attempt.json',{'fixture':True,'cycle_id':cycle.name})
    write(cycle/'rlt-stopped.json',{'fixture':True,'cycle_id':cycle.name})
elif a.action=='resume':
    assert a.release_receipt and a.release_receipt.resolve().is_relative_to(root.resolve())
    release=read(a.release_receipt)
    wm=read(a.release_receipt.parent/'wm-release.json')
    assert wm['fixture'] and wm['all_workers_stopped'] and wm['processes_clear'] and wm['gpus_released']
    assert release['cycle_id']==cycle.name and release['all_workers_stopped'] is True
    assert release['workload_route']=='WM_RLT_DIRECT'
    assert release['cleanup_receipt']['fixture'] is True
    assert release['cleanup_receipt']['processes_clear'] and release['cleanup_receipt']['gpus_released']
    # Exclusive files make duplicate resume calls fail, even in this fixture.
    write(cycle/'fixture-resume-intent.json',{'fixture':True,'release':str(a.release_receipt)})
    event('RLT_RESUME')
    write(cycle/'resumed-dispatched.json',{'fixture':True,'cycle_id':cycle.name})
else:
    assert (cycle/'resumed-dispatched.json').is_file()
    event('RLT_STATUS')
    print(json.dumps({'fixture':True,'cycle_id':cycle.name,'all_first_rounds_verified':True,
        'runs':{f'gpu{i}':{'fixture':True,'first_round_verified':True} for i in range(4,8)}}))
'''


DOJO_FIXTURE = r'''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--plan-only',action='store_true')
p.add_argument('--cleanup-only',action='store_true')
a=p.parse_args()
cfg=json.loads(a.config.read_text())
root=Path(cfg['fixture_root']);run=Path(cfg['fixture_evaluation_run'])
assert cfg['fixture'] is True and run.resolve().is_relative_to(root.resolve())
def event(kind):
    with (root/'fixture-events.jsonl').open('a') as f:
        f.write(json.dumps({'fixture':True,'event':kind,'time':time.time()})+'\n')
if a.plan_only:
    assert not a.cleanup_only
    event('DOJO_PLAN_ONLY')
elif a.cleanup_only:
    event('DOJO_CLEANUP_ONLY')
    # The only clear/released values in this helper describe these CPU fixtures.
    receipt={'fixture':True,'scope':'CPU test fixture, no real resource release',
             'processes_clear':True,'gpus_released':True,'time':time.time()}
    with (run/'cleanup-only-latest.json').open('x') as f:json.dump(receipt,f)
else:
    event('DOJO_EVALUATION_FORBIDDEN')
    raise SystemExit(97)
'''


def fixture(root, owner, outcome):
    project = root / 'project'
    evaluation = project / 'runs' / 'fixture-evaluation'
    prep = evaluation / 'fixture-preparation'
    attempt = evaluation / 'fixture-attempt'
    cycle = project / 'fixture-cycle'
    base = root / 'fixture-base'
    for directory in (prep, cycle, base):
        directory.mkdir(mode=0o700, parents=True)
    helper = cycle / 'rlt_cycle_sz3.py'
    helper.write_text(RLT_FIXTURE, encoding='utf-8')
    (base / 'dojo_sweep.py').write_text(DOJO_FIXTURE, encoding='utf-8')
    save(cycle / 'fixture.json', dict(fixture=True, root=str(root)))
    save(cycle / 'plan.json', dict(fixture=True, script_sha256=sha(helper)))
    cfg = dict(fixture=True, fixture_root=str(root), fixture_evaluation_run=str(evaluation),
               project=str(project), run_id=evaluation.name, sim_env=str(root / 'fixture-sim-env'))
    config = project / 'fixture-config.json'
    save(config, cfg)
    save(evaluation / 'plan.json', dict(fixture=True, config=cfg))
    save(prep / 'wm-spec.json', dict(fixture=True, outcome=outcome, root=str(root)))
    save(prep / 'ready.json', dict(
        fixture=True, config_path=str(config), config_sha256=sha(config),
        base_source_dir=str(base), cycle_dir=str(cycle), attempt_dir=str(attempt),
        prior_plan_sha256=sha(evaluation / 'plan.json'), wm_spec_sha256=sha(prep / 'wm-spec.json'),
        benchmark_unchanged=True, source_sha256={owner.name: sha(owner)},
        reuse_borrowed_cycle=False, return_rlt_direct=True, restore_rlt_after_dojo=True))
    return dict(root=str(root), owner=str(owner), config=str(config), cycle=str(cycle),
                prep=str(prep), attempt=str(attempt), base=str(base), project=str(project),
                evaluation=str(evaluation), outcome=outcome)


def child(fixture_path):
    """Import the real helpers, stub only workloads, then run the real owner."""
    account()
    data = read(fixture_path)
    root = Path(data['root'])
    owner = Path(data['owner'])
    assert root.resolve().is_relative_to(ACCOUNT_ROOT.resolve())
    assert root.stat().st_uid == ACCOUNT_UID and not root.is_symlink()
    common_path = owner.parent / 'common.py'
    spec = importlib.util.spec_from_file_location('common', common_path)
    common = importlib.util.module_from_spec(spec)
    sys.modules['common'] = common
    spec.loader.exec_module(common)
    common.account()

    def bounded(path):
        path = Path(path).absolute()
        assert path.resolve().is_relative_to(root.resolve()), 'Fixture path escaped temporary root'
        return path

    def load_base(path):
        assert bounded(path) == Path(data['base'])
        return Path(path), types.SimpleNamespace(gpu_processes=lambda physical: [])

    def bind_config(path):
        assert bounded(path) == Path(data['config'])
        cfg = read(path)
        assert cfg['fixture'] is True
        return Path(path), cfg, bounded(data['project']), bounded(data['evaluation'])

    def event(kind):
        with (root / 'fixture-events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(fixture=True, event=kind, time=time.time())) + '\n')

    def run_stage(wm_spec, cycle, attempt, gpu_processes, state, stop_requested):
        assert wm_spec['fixture'] is True and bounded(wm_spec['root']) == root
        assert bounded(cycle) == Path(data['cycle']) and (cycle / 'rlt-stopped.json').is_file()
        assert bounded(attempt) == Path(data['attempt'])
        assert not gpu_processes([4, 5, 6, 7]) and not stop_requested()
        state('RUNNING_WM', fixture=True)
        event('WM_FIXTURE_RELEASED')
        released = dict(fixture=True, scope=FIXTURE_SCOPE, time=time.time(),
            cycle_id=cycle.name, owner_token='fixture-not-a-real-owner',
            outcome=wm_spec['outcome'], wm_exit_code=0 if wm_spec['outcome'] == 'completed' else 17,
            error=None if wm_spec['outcome'] == 'completed' else 'fixture WM failure',
            physical_gpus=[4, 5, 6, 7], managed_processes=[],
            all_workers_stopped=True, processes_clear=True, gpus_released=True)
        common.atomic(attempt / 'wm-release.json', released)
        return released

    guard = types.ModuleType('process_guard')
    guard.error_context = lambda exc: dict(fixture=True, exception=type(exc).__name__, error=repr(exc))
    watchdog = types.ModuleType('hang_watchdog')
    watchdog.identity = common.identity
    wm = types.ModuleType('wm_stage')
    wm.run_stage = run_stage

    original_popen = subprocess.Popen
    approved = {Path(data['cycle']) / 'rlt_cycle_sz3.py', Path(data['base']) / 'dojo_sweep.py'}

    def fixture_popen(argv, *args, **kwargs):
        argv = list(argv)
        assert len(argv) >= 4 and argv[1:3] == ['-u', '-B']
        script = bounded(argv[3])
        assert script in approved, 'Only the two explicit CPU fixture helpers may execute'
        assert argv[0] in (str(Path(read(data['config'])['sim_env']) / 'bin/python'),
                           '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python')
        argv[0] = sys.executable
        # Keep the owner's real subprocess I/O, wait, and exit handling.
        return original_popen(argv, *args, **kwargs)

    argv = [str(owner), '--config', data['config'], '--cycle-dir', data['cycle'],
            '--preparation-dir', data['prep']]
    with patch.object(common, 'load_base', load_base), patch.object(common, 'bind_config', bind_config), \
         patch.dict(sys.modules, process_guard=guard, hang_watchdog=watchdog, wm_stage=wm), \
         patch.object(subprocess, 'Popen', fixture_popen), patch.object(sys, 'argv', argv):
        # SystemExit propagates, so the parent verifies the actual owner exit code.
        runpy.run_path(str(owner), run_name='__main__')


class DirectReturnChecks(unittest.TestCase):
    owner = None
    reports = []

    def check_outcome(self, outcome):
        with tempfile.TemporaryDirectory(prefix='wm-direct-return-cpu-', dir=ACCOUNT_ROOT) as temporary:
            root = Path(temporary)
            data = fixture(root, self.owner, outcome)
            fixture_path = root / 'fixture-input.json'
            save(fixture_path, data)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                '--child-fixture', str(fixture_path)], env=env, capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0 if outcome == 'completed' else 1,
                             (result.stdout + '\n' + result.stderr)[-4000:])
            attempt = Path(data['attempt'])
            final = read(attempt / 'pipeline-final.json')
            release = read(attempt / 'dojo-release.json')
            events = [json.loads(line)['event'] for line in (root / 'fixture-events.jsonl').read_text().splitlines()]
            phases = [json.loads(line)['phase'] for line in (attempt / 'pipeline-events.jsonl').read_text().splitlines()]
            self.assertEqual(events, ['DOJO_PLAN_ONLY', 'RLT_STOP', 'WM_FIXTURE_RELEASED',
                                     'DOJO_CLEANUP_ONLY', 'RLT_RESUME', 'RLT_STATUS'])
            self.assertEqual(events.count('RLT_RESUME'), 1)
            self.assertNotIn('EVALUATING', phases)
            self.assertNotIn('WM_RELEASED_RESUMING_DOJO', phases)
            self.assertIn('WM_RELEASED_RETURNING_RLT_DIRECT', phases)
            self.assertEqual(final['terminal_status'], outcome)
            self.assertIsNone(final['error'])
            self.assertIsNone(final['dojo_exit_code'])
            self.assertTrue(final['wm_attempted'] and final['wm_released'] and final['rlt_dispatched'])
            self.assertEqual(final['rlt_first_round']['state'], 'verified')
            self.assertTrue(final['rlt_first_round']['status']['fixture'])
            self.assertEqual(release['workload_route'], 'WM_RLT_DIRECT')
            self.assertTrue(release['all_workers_stopped'] and release['cleanup_receipt']['fixture'])
            for label in ('final-preflight', 'rlt-stop', 'dojo-cleanup', 'rlt-resume'):
                self.assertEqual(read(attempt / (label + '-exit.json'))['exit_code'], 0)
            self.assertFalse((attempt / 'dojo-controller-identity.json').exists())
            self.reports.append(dict(fixture=True, outcome=outcome, owner_exit_code=result.returncode,
                events=events, phases=phases, exactly_one_fixture_resume=True,
                no_dojo_evaluation=True, verified_fixture_return=True))

    def test_completed_wm_returns_rlt_once_without_dojo_evaluation(self):
        self.check_outcome('completed')

    def test_failed_wm_returns_rlt_once_without_dojo_evaluation(self):
        self.check_outcome('failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-script', type=Path,
        default=ACCOUNT_ROOT / 'projects/wan-goal-sz3/scripts/resource_switch/continue_pipeline.py')
    parser.add_argument('--child-fixture', type=Path)
    args = parser.parse_args()
    account()
    if args.child_fixture:
        child(args.child_fixture)
        return
    owner = args.owner_script.absolute()
    if not owner.is_file() or owner.stat().st_uid != ACCOUNT_UID:
        raise RuntimeError('An existing account-owned continue_pipeline.py is required')
    DirectReturnChecks.owner = owner
    started = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DirectReturnChecks))
    print(json.dumps(dict(time=time.time(), ok=result.wasSuccessful(), fixture=True,
        scope=FIXTURE_SCOPE, owner_script=str(owner), owner_sha256=sha(owner),
        tests_run=result.testsRun, seconds=time.time() - started, cases=DirectReturnChecks.reports)))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
