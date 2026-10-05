"""CPU checks for the single-card boundary; run on SZ3 before borrowing."""
import copy
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

import rynn_diagnostic_owner as O


def fixture():
    argv = [O.RYNN_PYTHON, '-u', '-B', str(O.D / 'code/rynn_diagnostic.py'),
        '--service-module', str(O.S / 'rynn-control-v2/code/rynn_success_service.py'),
        '--model-path', str(O.MODEL), '--manifest-path', str(O.MODEL / 'manifest.json'),
        '--physical-gpu', '4', '--cases-json', str(O.D / 'prepared/cases.json'),
        '--samples-npz', str(O.D / 'prepared/samples.npz'), '--output', str(O.O / 'result.json')]
    return dict(mode='rynn-single-gpu-diagnostic', physical_gpus=[4], untouched_gpus=[5, 6, 7],
        owner_dir=str(O.O), parent_owner=str(O.PARENT), timeout_seconds=900, argv=argv,
        environment={'CUDA_VISIBLE_DEVICES': '4', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'},
        untouched_drivers={key: {} for key in ('gpu5', 'gpu6', 'gpu7')})


class TestDiagnosticOwner(unittest.TestCase):
    def test_exact_child_contract(self):
        O.validate_contract(fixture())
        for field, value in [('physical_gpus', [4, 5]), ('timeout_seconds', 1800)]:
            plan = fixture()
            plan[field] = value
            with self.assertRaises(AssertionError):
                O.validate_contract(plan)
        plan = fixture()
        plan['argv'] += ['--extra-child', 'bad']
        with self.assertRaises(AssertionError):
            O.validate_contract(plan)

    def test_mask_cannot_expand(self):
        plan = fixture()
        plan['environment']['CUDA_VISIBLE_DEVICES'] = '4,5'
        with self.assertRaises(AssertionError):
            O.validate_contract(plan)

    def test_gpu_scope_ignores_other_jobs(self):
        catalog = types.SimpleNamespace(scan=Mock(), live=lambda: [{'pid': 10}])
        H = types.SimpleNamespace(gpu_processes=lambda _g: [{'pid': 10, 'gpu': 4}, {'pid': 99, 'gpu': 5}])
        self.assertEqual(len(O.check_gpu_scope(H, catalog)), 2)

    def test_gpu_scope_rejects_escape_or_foreign(self):
        catalog = types.SimpleNamespace(scan=Mock(), live=lambda: [{'pid': 10}])
        for context in ({'pid': 10, 'gpu': 0}, {'pid': 99, 'gpu': 4}):
            H = types.SimpleNamespace(gpu_processes=lambda _g: [context])
            with self.assertRaises(AssertionError):
                O.check_gpu_scope(H, catalog)

    def test_cleanup_only_exact_catalog_targets(self):
        target = {'pid': 10, 'start': 100, 'uid': 20001}
        alive = [target]
        signals = []
        def send(row, sig):
            signals.append((row, sig))
            alive.clear()
        catalog = types.SimpleNamespace(scan=Mock(), live=lambda: list(alive), send=send,
            rows={(10, 100): target})
        result = O.cleanup(None, catalog)
        self.assertTrue(result['all_stopped'])
        self.assertEqual([row for row, _ in signals], [target])


if __name__ == '__main__':
    unittest.main()
