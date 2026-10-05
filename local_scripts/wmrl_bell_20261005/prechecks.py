"""Finite prechecks inside the existing four-card owner, before smoke/formal.

No new owner, borrowing, adoption or daemon. launch/catalog/cleanup and the
original owner's finally retain ownership of every process and Ray namespace.
"""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import time

ORDER = [('native_rynn32', 'native'), ('rynn32', 'rynn_binary'),
         ('native_bell32', 'native'), ('bell_rm', 'bell_reward')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scoped_plan(plan):
    """Expose native prechecks to existing exact namespace cleanup/allowlisting."""
    result = dict(plan)
    result['trials'] = list(plan['trials']) + [row for row in plan.get('prechecks', []) if row['kind'] == 'native']
    return result


def validate_prechecks(plan, M):
    rows = plan['prechecks']
    assert [(r['key'], r['kind']) for r in rows] == ORDER
    owner = Path(plan['owner_dir'])
    keys = [r['key'] for r in rows] + [r['key'] for r in plan['trials']]
    assert len(keys) == len(set(keys))
    namespaces = [r['namespace'] for r in rows if r['kind'] == 'native'] + [r['namespace'] for r in plan['trials']]
    assert len(namespaces) == len(set(namespaces))
    for row in rows:
        assert re.fullmatch(r'[a-z][a-z0-9_]+', row['key'])
        assert 0 < row['timeout_seconds'] <= 7200
        assert 1 <= len(row['commands']) <= 3
        if row['kind'] == 'native':
            assert row['namespace'].startswith('opendw_')
            assert row['commands'][0]['mode'] == 'ray'
            assert all(c['mode'] == 'cpu' for c in row['commands'][1:])
        else:
            assert 'namespace' not in row and row['commands'][0]['mode'] == 'gpu'
            output = Path(row['result_file'])
            assert output.is_absolute() and output.resolve().is_relative_to((owner / row['key']).resolve())
        for index, command in enumerate(row['commands']):
            argv = command['argv']
            assert isinstance(argv, list) and len(argv) >= 4 and all(isinstance(arg, str) for arg in argv)
            assert argv[1:3] == ['-u', '-B'], 'Use a direct, unbuffered Python child'
            assert Path(argv[0]).is_absolute() and Path(argv[0]).is_file()
            source = M.owned_path(argv[3])
            assert plan['source_sha256'][str(source)] == sha(source)
            M.owned_path(command['cwd'])
            assert command['mode'] in ('cpu', 'ray', 'gpu')
            if command['mode'] == 'gpu':
                assert command['physical_gpu'] == 4
            if command.get('environment_file'):
                path = M.owned_path(command['environment_file'])
                assert plan['source_sha256'][str(path)] == sha(path)
            if command['mode'] == 'ray':
                assert index == 0 and row['kind'] == 'native'
                assert argv.count('--receipt-dir') == argv.count('--namespace') == argv.count('--config') == 1
                assert argv[argv.index('--receipt-dir') + 1] == str(owner / row['key'])
                assert argv[argv.index('--namespace') + 1] == row['namespace']
                config = M.owned_path(argv[argv.index('--config') + 1])
                assert plan['source_sha256'][str(config)] == sha(config)
            assert not any(key in command.get('environment', {}) for key in M.MASKS)


def bell_decision(report):
    """Check the fixed 0.9 classifier against native positive/negative labels.

This is a small sanity check, not a fitted threshold or proof of WM reliability.
The sample builder must exclude post-success latched frames from this matrix.
"""
    if report.get('strict_load') is not True or not report.get('rows'):
        raise RuntimeError('Bell classifier did not produce strict-loaded finite sample scores')
    row = report['thresholds']['0.9']
    tp, fn, fp, tn = [int(row[k]) for k in ('true_positive', 'false_negative', 'false_positive', 'true_negative')]
    positive, negative = tp + fn, fp + tn
    if not positive or not negative:
        return dict(passed=False, reason='Native batch lacks positive or negative labels; separation not established',
                    positives=positive, negatives=negative, threshold=0.9)
    sensitivity, specificity = tp / positive, tn / negative
    # No threshold search: a reversed/constant/chance-or-worse result cannot
    # support the intended successful=1 / unsuccessful=0 reward.
    balanced_accuracy = (sensitivity + specificity) / 2
    passed = tp > 0 and tn > 0 and balanced_accuracy > 0.5
    return dict(passed=passed, reason='Native binary sanity passed' if passed else 'Native success/failure not distinguished',
                threshold=0.9, positives=positive, negatives=negative,
                true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn,
                balanced_accuracy=balanced_accuracy, transfer_to_generated_frames_verified=False)


def run_prechecks(plan, catalog, launch, M):
    owner = Path(plan['owner_dir'])
    completed = []
    for row in plan['prechecks']:
        target = owner / row['key']
        target.mkdir(mode=0o700, exist_ok=False)
        start = time.monotonic()
        if row['kind'] == 'native':
            assert not M.H.active(M.H.actors(plan), row['namespace']), 'Precheck namespace already occupied'
        records = []
        for index, command in enumerate(row['commands']):
            env = dict(M.read(plan['environment_file']))
            if command.get('environment_file'):
                env.update(M.read(command['environment_file']))
            env.update(command.get('environment', {}))
            for key in M.MASKS:
                env.pop(key, None)
            env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            if command['mode'] == 'gpu':
                env['CUDA_VISIBLE_DEVICES'] = str(command['physical_gpu'])
            elif command['mode'] == 'cpu':
                env['CUDA_VISIBLE_DEVICES'] = ''
            else:
                env['RAY_ADDRESS'] = plan['ray_address']
            child = launch(command['argv'], command['cwd'], env, row['key'], target / ('command-' + str(index) + '.log'))
            while child.poll() is None:
                if row['kind'] == 'native':
                    M.register_actors(plan, row, catalog)
                M.resource_snapshot(plan, catalog, row['key'])
                elapsed = time.monotonic() - start
                if elapsed >= row['timeout_seconds']:
                    raise TimeoutError('Finite precheck deadline: ' + row['key'])
                M.atomic(owner / 'state.json', dict(time=M.H.now(), phase=row['key'], command=index, elapsed_seconds=elapsed))
                time.sleep(5)
            # Catch Ray actors that were created immediately before driver exit.
            if row['kind'] == 'native':
                M.register_actors(plan, row, catalog)
            cleanup = M.cleanup(plan, catalog, row['key'])
            M.record(target / ('cleanup-' + str(index) + '.json'), cleanup)
            status = dict(command=index, exit_code=child.returncode, elapsed_seconds=time.monotonic() - start)
            records.append(status)
            M.record(target / ('command-' + str(index) + '.json'), status)
            if child.returncode != 0:
                raise RuntimeError('Finite precheck process failed: ' + row['key'] + '/' + str(index))
            if command['mode'] == 'ray':
                receipt = M.read(target / 'ray-job.json')
                assert receipt['namespace'] == row['namespace'] and receipt['job_id']
        result = dict(key=row['key'], engineering_passed=True, commands=records,
                      seconds=time.monotonic() - start)
        if row['kind'] != 'native':
            report = M.read(row['result_file'])
            result['result_file'] = row['result_file']
            result['result_sha256'] = sha(row['result_file'])
            if row['kind'] == 'rynn_binary':
                assert report['engineering_passed'] is True, 'Rynn inference did not finish'
                result['reward_gate_blocks_bell'] = False
                result['note'] = 'Rynn semantic result is recorded only; continue the separately selected bell reward.'
            else:
                result['reward_gate'] = bell_decision(report)
        M.record(target / 'precheck.json', result)
        completed.append(result)
        if row['kind'] == 'bell_reward' and not result['reward_gate']['passed']:
            M.record(owner / 'prechecks.json', dict(passed=False, completed=completed,
                reason=result['reward_gate']['reason']))
            raise RuntimeError('Bell reward precheck: ' + result['reward_gate']['reason'])
    M.record(owner / 'prechecks.json', dict(passed=True, completed=completed,
        no_learning_gradient_gate_added=True))


def install(M, plan, owner_main_source=None):
    """Call after the existing formal wrapper installs M; return that same module."""
    assert not plan.get('borrow_adoption'), 'Use normal fresh borrowing, not an adoption layer'
    assert not getattr(M, '_finite_prechecks_installed', False), 'Do not patch twice'
    source = owner_main_source if owner_main_source is not None else inspect.getsource(M.owner_main)
    anchor = "        for row in plan['trials']:\n            env = dict(base_env)"
    assert source.count(anchor) == 1
    replacement = source.replace(anchor,
        "        run_finite_prechecks(plan, catalog, launch)\n" + anchor, 1)
    original_cleanup, original_allowlist, original_validate, original_record = M.cleanup, M.add_allowlist, M.validate, M.record

    def cleanup(current, catalog, phase=None):
        return original_cleanup(scoped_plan(current), catalog, phase)

    def add_allowlist(current):
        return original_allowlist(scoped_plan(current))

    def validate(current, frozen=False):
        value = original_validate(current, frozen)
        validate_prechecks(current, M)
        return value

    delta = dict(kind='finite-prechecks-before-original-trials',
        hook_sha256=sha(__file__),
        before_sha256=hashlib.sha256(source.encode()).hexdigest(),
        after_sha256=hashlib.sha256(replacement.encode()).hexdigest(),
        borrowing_and_finally_unchanged=True)

    def record(path, value):
        original_record(path, value)
        if Path(path).name == 'owner-plan.json':
            original_record(Path(path).parent / 'prechecks-source-delta.json', delta)

    M.cleanup, M.add_allowlist, M.validate, M.record = cleanup, add_allowlist, validate, record
    M.run_finite_prechecks = lambda current, catalog, launch: run_prechecks(current, catalog, launch, M)
    exec(compile(replacement, str(Path(__file__)) + ':finite-prechecks-owner', 'exec'), M.__dict__)
    M._finite_prechecks_installed = True
    return M
