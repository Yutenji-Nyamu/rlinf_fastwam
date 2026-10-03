"""CPU-only tests for the new return scope; no Ray connection/GPU calls/subprocesses."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('opendw_smoke_gpu4_cycle.py')
spec = importlib.util.spec_from_file_location('cycle_graphics_test_target', SOURCE)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.runtime = self.root/'returned/runtime'
        self.runtime.mkdir(parents=True)
        self.receipts = self.root/'receipts'
        self.receipts.mkdir()
        self.manifest = dict(gpu_uuid='GPU-physical4', commname='odw4-test000000', token='scope-test',
                             pci='00000000:96:00.0', home='/home/chenyiteng', title_api='ray._raylet.setproctitle',
                             marker_path='/unique/scope-marker.so', receipts_dir=str(self.receipts))
        self.scope = dict(manifest_path=str(self.root/'scope.json'))
        self.plan = dict(graphics_scope=self.scope, runs={'gpu4': dict(
            new_run=str(self.runtime.parent), namespace='return-test')})
        self.driver = dict(pid=100, uid=20001, start=10)
        self.tree = {100: self.driver, 200: dict(pid=200, uid=20001, start=20)}
        self.gpus = [dict(gpu=4, pid=200, type='C+G')]
        self.events = []
        self.fragment = {'RLINF_OPENDW_GPU_SCOPE_MANIFEST': self.scope['manifest_path'],
                         'LD_PRELOAD': '/unique/scope-marker.so', 'PYTHONPATH': '/unique/bootstrap',
                         '__GL_APPLICATION_PROFILE': '1'}
        self.scope_helper = types.SimpleNamespace(environment_fragment=lambda *_: dict(self.fragment))
        self.original_status = lambda _: {'all_first_rounds_verified': True,
                                          'runs': {'gpu4': {'first_round_verified': True}}}
        self.h = types.SimpleNamespace(
            MASKS=('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'),
            driver=lambda *_: None, status=self.original_status,
            now=lambda: 'test-time', proc=lambda _: self.driver, same=lambda _: True,
            save=lambda p, v: self.events.append((str(p), v)),
            actors=lambda _: [dict(pid=200, job_id='job1')], active=lambda rows, _: rows,
            validate_actor_rows=lambda *_: None, process_tree=lambda _: self.tree,
            gpu_processes=lambda _: self.gpus)
        self.save(self.runtime/'driver-identity.json', self.driver)
        self.save(self.runtime/'graphics-scope-job.json', dict(namespace='return-test', job_id='job1',
                  manifest_path=self.scope['manifest_path'], driver=self.driver))
        for pid, ident in self.tree.items():
            self.save(self.receipts/f'{pid}-bootstrap.json', dict(pid=pid, token='scope-test',
                phase='bootstrap', comm=self.manifest['commname'], cuda_visible_devices='4',
                start_ticks=ident['start'], gpu_uuid='GPU-physical4', home='/home/chenyiteng',
                setproctitle_patched=True, setproctitle_api='ray._raylet.setproctitle'))
        self.save(self.receipts/'200-renderer.json', dict(pid=200, token='scope-test',
            phase='renderer_bound', comm=self.manifest['commname'], start_ticks=20,
            cuda_visible_devices='4', cuda_uuid='GPU-physical4', cuda_device_count=1, cuda_pci='0000:96:00.0'))
        self.start_patch(patch.object(C, 'H', self.h))
        self.start_patch(patch.object(C, 'load_graphics_scope', return_value=(self.scope_helper, self.manifest)))
        original_read = Path.read_text

        def proc_read(path, *args, **kwargs):
            if str(path).startswith('/proc/'):
                if path.name == 'comm':
                    return self.manifest['commname']
                if path.name == 'maps':
                    return '1000-2000 r-xp 0000 00:00 0 /unique/scope-marker.so\n'
                raise AssertionError('Unexpected process read: '+str(path))
            return original_read(path, *args, **kwargs)

        self.start_patch(patch.object(Path, 'read_text', proc_read))

    def start_patch(self, value):
        value.start()
        self.addCleanup(value.stop)

    @staticmethod
    def save(path, value):
        path.write_text(json.dumps(value))

    def test_valid_scope_requires_live_marker_bootstrap_and_renderer(self):
        result = C.return_graphics_status(self.plan)
        self.assertTrue(result['verified'])
        self.assertTrue(result['scope_inherited'])

    def test_owned_gpu0_context_fails(self):
        self.gpus.append(dict(gpu=0, pid=200, type='G'))
        result = C.return_graphics_status(self.plan)
        self.assertFalse(result['verified'])
        self.assertFalse(result['all_owned_contexts_gpu4'])

    def test_unrelated_historical_gpu0_does_not_fail_new_job(self):
        self.gpus.append(dict(gpu=0, pid=999, type='G'))
        result = C.return_graphics_status(self.plan)
        self.assertTrue(result['verified'])
        self.assertEqual([r['pid'] for r in result['owned_contexts']], [200])

    def test_missing_renderer_proof_fails(self):
        (self.receipts/'200-renderer.json').unlink()
        self.assertFalse(C.return_graphics_status(self.plan)['scope_inherited'])

    def test_nonnumeric_cuda_placement_fails(self):
        path = self.receipts/'200-bootstrap.json'
        value = json.loads(path.read_text())
        value['cuda_visible_devices'] = 'GPU-wrong'
        self.save(path, value)
        self.assertFalse(C.return_graphics_status(self.plan)['verified'])

    def test_recycled_pid_receipt_fails(self):
        path = self.receipts/'200-bootstrap.json'
        value = json.loads(path.read_text())
        value['start_ticks'] = 999
        self.save(path, value)
        self.assertFalse(C.return_graphics_status(self.plan)['verified'])

    def test_wrong_renderer_physical_identity_fails(self):
        path = self.receipts/'200-renderer.json'
        value = json.loads(path.read_text())
        value['cuda_pci'] = '0000:17:00.0'
        self.save(path, value)
        self.assertFalse(C.return_graphics_status(self.plan)['verified'])

    def install(self, driver):
        self.h.driver = driver
        self.start_patch(patch.object(C, 'import_file', return_value=self.h))
        self.start_patch(patch.object(C, 'load_plan', return_value=self.plan))
        C.install_helper(self.root, '/unused/helper.py')

    def test_failed_scope_blocks_first_round_receipt(self):
        self.install(lambda *_: None)
        self.gpus.append(dict(gpu=0, pid=200, type='G'))
        status = C.H.status(self.root)
        self.assertFalse(status['all_first_rounds_verified'])
        self.assertFalse(status['runs']['gpu4']['first_round_verified'])

    def test_driver_injects_only_job_fragment_and_preserves_runtime(self):
        calls = []
        ray = types.SimpleNamespace(init=lambda *a, **kw: calls.append((a, kw)),
                                   get_runtime_context=lambda: types.SimpleNamespace(get_job_id=lambda: 'job1'))
        original_init = ray.init
        self.start_patch(patch.dict('sys.modules', {'ray': ray}))
        (self.runtime/'graphics-scope-job.json').unlink()
        self.install(lambda *_: ray.init(address='auto', namespace='return-test',
                     runtime_env={'env_vars': {'KEEP': 'original'}, 'config': {'setup_timeout_seconds': 30}}))
        with patch.dict(os.environ, {'RLINF_OPENDW_GPU_SCOPE_MANIFEST': self.scope['manifest_path']}, clear=True):
            C.H.driver(self.root, 'gpu4')
        runtime = calls[0][1]['runtime_env']
        self.assertEqual(runtime['env_vars']['KEEP'], 'original')
        self.assertEqual(runtime['config']['setup_timeout_seconds'], 30)
        self.assertEqual(runtime['env_vars']['LD_PRELOAD'], self.fragment['LD_PRELOAD'])
        self.assertNotIn('CUDA_VISIBLE_DEVICES', runtime['env_vars'])
        self.assertIs(ray.init, original_init)
        self.assertEqual(self.events[-1][1]['job_id'], 'job1')

    def test_job_level_cuda_mask_is_rejected(self):
        calls = []
        ray = types.SimpleNamespace(init=lambda *a, **kw: calls.append((a, kw)))
        self.start_patch(patch.dict('sys.modules', {'ray': ray}))
        self.install(lambda *_: ray.init(runtime_env={'env_vars': {'CUDA_VISIBLE_DEVICES': '4'}}))
        with patch.dict(os.environ, {'RLINF_OPENDW_GPU_SCOPE_MANIFEST': self.scope['manifest_path']}, clear=True):
            with self.assertRaisesRegex(AssertionError, 'CPU/discovery'):
                C.H.driver(self.root, 'gpu4')
        self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
