"""CPU-only regression checks; no CUDA, Ray, subprocess or live operations."""
import os
from pathlib import Path
import sys
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from lease_common import environment
from prepare_lease import replace_outputs
from gpu_scope_runtime import check_cuda
from capture_rlt67 import protected_gpu
from stop_handoff import protect


class LeaseCPU(unittest.TestCase):
    def test_probe_mask_never_leaks_to_formal_or_owner(self):
        plan = {'slots': {'6': {'scope_env': {'LD_PRELOAD': '/new/scope.so',
                                               'PYTHONPATH': '/new/bootstrap:/new/repo'}}}}
        original = {'PATH': '/usr/bin', 'CUDA_VISIBLE_DEVICES': '0,1',
                    'LD_PRELOAD': '/old/scope.so', 'RLINF_OPENDW_GPU_SCOPE_MANIFEST': '/old/scope'}
        with patch.dict(os.environ, original, clear=True):
            probe = environment(plan, 6, probe=True)
            formal = environment(plan, 6, probe=False)
            self.assertEqual(probe['CUDA_VISIBLE_DEVICES'], '6')
            self.assertNotIn('CUDA_VISIBLE_DEVICES', formal)
            self.assertEqual(dict(os.environ), original)
            self.assertEqual(formal['LD_PRELOAD'], '/new/scope.so')

    def test_resume_output_rewrite_preserves_method_budget(self):
        value = {'runner': {'max_steps': 3000, 'resume_dir': None},
                 'env': {'train': {'total_num_envs': 8, 'max_steps_per_rollout_epoch': 200}},
                 'path': '/runs/old/video', 'name': 'old', 'task': 'click_bell'}
        changed = replace_outputs(value, Path('/runs/old'), Path('/runs/new'))
        self.assertEqual(changed['runner'], value['runner'])
        self.assertEqual(changed['env'], value['env'])
        self.assertEqual(changed['path'], '/runs/new/video')
        self.assertEqual(changed['name'], 'new')
        self.assertEqual(value['path'], '/runs/old/video')

    def test_native_renderer_rejects_other_physical_gpu(self):
        manifest = {'physical_gpu': 6, 'gpu_uuid': 'GPU-test'}
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '7'}):
            with self.assertRaises(RuntimeError):
                check_cuda(manifest, require=True)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '6'}):
            self.assertEqual(check_cuda(manifest, require=True), '6')

    def test_cpu_inventory_can_be_unmasked_but_renderer_cannot(self):
        manifest = {'physical_gpu': 6, 'gpu_uuid': 'GPU-test'}
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(check_cuda(manifest))
            with self.assertRaises(RuntimeError):
                check_cuda(manifest, require=True)

    def test_finished_historical_protected_job_allows_new_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'oldrun/runtime'
            runtime.mkdir(parents=True)
            for path in (root / 'finished-place_fan-clean.json', runtime / 'finished.json'):
                path.write_text(json.dumps({'exit_code': -6}))
            old = {'pid': 11, 'uid': 1003, 'start': 22}
            b = SimpleNamespace(proc=lambda pid: None, same=lambda identity: False)
            snapshots = [{'uuid': 'unused'}] * 8
            snapshots[4] = {'uuid': 'GPU-four', 'processes': [{'pid': 999, 'type': 'C'}]}
            row = {'gpus': [4], 'run': str(root / 'oldrun'), 'task': 'place_fan',
                   'role': 'clean', 'namespace': 'old-place-fan'}
            result = protected_gpu(b, root, {'roles': {'place_fan/clean': 'FAILED'}}, row, old, snapshots)
            self.assertEqual(result['historical_state'], 'TERMINAL')
            self.assertEqual(len(result['terminal_proof']), 2)
            self.assertNotIn('999', json.dumps(result))

    def test_owner_natural_end_requires_matching_final_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = {'pid': 11, 'uid': 1003, 'start': 22}
            state = {'owner': old, 'time': 0, 'roles': {'a': 'FAILED', 'b': 'COMPLETE'}}
            (root / 'queue-status.json').write_text(json.dumps(state))
            b = SimpleNamespace(proc=lambda pid: None, same=lambda identity: False)
            capture = {'stage': str(root), 'owner': old, 'protected': {}}
            with self.assertRaises(AssertionError):
                protect(b, capture, allow_owner_terminal=True)
            (root / 'queue-finished.json').write_text(json.dumps({**state, 'retained_drivers': {}}))
            protect(b, capture, allow_owner_terminal=True)
            with self.assertRaises(AssertionError):
                protect(b, capture, allow_owner_terminal=False)

    def test_reused_owner_pid_is_always_rejected(self):
        old = {'pid': 11, 'uid': 1003, 'start': 22}
        b = SimpleNamespace(proc=lambda pid: {**old, 'start': 23}, same=lambda identity: False)
        with self.assertRaises(AssertionError):
            protect(b, {'owner': old}, allow_owner_terminal=True)


if __name__ == '__main__':
    unittest.main()
