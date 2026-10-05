"""Reuse the single-GPU boundaries; add the one-card parent provenance case."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rynn_diagnostic_owner_v2 as O
import test_rynn_diagnostic_owner as original


def fixture():
    argv = [O.RYNN_PYTHON, '-u', '-B', str(O.D / 'code/rynn_native_diagnostic.py'),
        '--service-module', str(O.S / 'rynn-control-v2/code/rynn_success_service.py'),
        '--model-path', str(O.MODEL), '--manifest-path', str(O.MODEL / 'manifest.json'),
        '--physical-gpu', '4', '--cases-json', str(O.CASES), '--controls-json', str(O.CONTROLS),
        '--native-frames-npz', str(O.NATIVE_FRAMES), '--native-frames-json', str(O.NATIVE_FRAMES_JSON),
        '--samples-npz', str(O.SAMPLES),
        '--official-inference', str(O.OFFICIAL), '--output', str(O.O / 'result.json')]
    return dict(mode='rynn-single-gpu-diagnostic', physical_gpus=[4], untouched_gpus=[5, 6, 7],
        owner_dir=str(O.O), parent_owner=str(O.PARENT), timeout_seconds=900, argv=argv,
        environment={'CUDA_VISIBLE_DEVICES': '4', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'},
        untouched_drivers={key: {} for key in ('gpu5', 'gpu6', 'gpu7')})


original.O, original.fixture = O, fixture
TestDiagnosticOwner = original.TestDiagnosticOwner


class TestSingleParent(unittest.TestCase):
    def test_single_parent_chain_and_wrong_gpu_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent, stage = root / 'parent', root / 'child'
            parent.mkdir()
            stage.mkdir()
            module = stage / 'rlt_returned_cycle.py'
            module.write_text('# frozen test helper\n')
            def write(path, value):
                path.write_text(json.dumps(value))
            write(stage / 'plan.json', dict(group='gpu4', cycle_id=stage.name,
                runs={'gpu4': {'gpus': [4], 'new_run': '/test/returned-gpu4'}}))
            identity = dict(pid=1234, uid=20001, start=900)
            write(parent / 'owner-identity.json', identity)
            write(parent / 'cleanup.json', dict(all_stopped=True))
            final = dict(terminal_status='completed', error=None, recovery_error=None,
                physical_gpus=[4], untouched_gpus=[5, 6, 7], lifecycle_path=str(stage),
                rlt_borrowed=True, rlt_return_dispatched=True)
            write(parent / 'final.json', final)
            release_path = parent / 'diagnostic-release.json'
            release = dict(cycle_id=stage.name, gpus=[4], terminal_status='completed',
                all_workers_stopped=True, managed_processes=[identity])
            write(release_path, release)
            verified = dict(path=str(release_path), sha256=O.sha(release_path), terminal_status='completed')
            write(stage / 'dojo-release-verified.json', verified)
            write(stage / 'resumed-dispatched.json', dict(cycle_id=stage.name, release=verified,
                runs={'gpu4': {'run': '/test/returned-gpu4'}}))
            write(parent / 'rlt-return-dispatched.json', dict(lifecycle_path=str(stage),
                result={'resumed_dispatched': True, 'cycle_id': stage.name}))
            write(parent / 'owner-plan.json', dict(mode='rynn-single-gpu-diagnostic', owner_dir=str(parent),
                physical_gpus=[4], untouched_gpus=[5, 6, 7], lifecycle_path=str(stage), lifecycle_module=str(module),
                source_sha256={str(stage / 'plan.json'): O.sha(stage / 'plan.json'), str(module): O.sha(module)}))
            with patch.object(O, 'PARENT', parent):
                key, returned_identity, evidence = O.single_parent_complete(parent, stage, module)
                self.assertEqual(key, 'gpu4')
                self.assertEqual(returned_identity, identity)
                self.assertEqual(evidence['combined_plan'], str(stage / 'plan.json'))
                release['gpus'] = [4, 5, 6, 7]
                write(release_path, release)
                with self.assertRaises(AssertionError):
                    O.single_parent_complete(parent, stage, module)


if __name__ == '__main__':
    unittest.main()
