"""Server CPU fixtures for the repaired GPU4 provenance and short run names."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('bell_prepare_fixture', Path(__file__).with_name('prepare_cycles.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PrepareCycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(patch.stopall)
        self.root = Path(self.tmp.name)
        self.s = self.root / 'project'
        self.owner = self.s / 'rynn-numeric-v1/run'
        self.repair = self.s / 'rynn-numeric-v1/rlt-return-repair-v2'
        self.stage = self.repair / 'rynn-numeric-v1-gpu4'
        self.original = self.s / 'rynn-numeric-v1/prepared/rynn-numeric-v1-gpu4'
        for path in (self.owner, self.stage, self.original):
            path.mkdir(parents=True)
        patch.multiple(M, ROOT=self.root, S=self.s, NUMERIC_OWNER=self.owner,
            REPAIR=self.repair, GPU4_STAGE=self.stage).start()
        patch.object(M, 'owned', side_effect=lambda p, exists=True: Path(p)).start()
        self.module = self.stage / 'rlt_returned_cycle.py'
        self.module.write_text('# frozen lifecycle\n')
        patch.object(M, 'CHILD_SHA', M.sha(self.module)).start()
        original_module = self.original / self.module.name
        original_module.write_bytes(self.module.read_bytes())
        self.save(self.original / 'plan.json', {'fixture': 'original'})
        identity = dict(pid=100, uid=20001, start=300)
        self.save(self.owner / 'owner-identity.json', identity)
        self.save(self.owner / 'owner-plan.json', dict(owner_dir=str(self.owner),
            mode='rynn-single-gpu-diagnostic', physical_gpus=[4], untouched_gpus=[5, 6, 7],
            lifecycle_path=str(self.original), lifecycle_module=str(original_module),
            source_sha256={str(self.original / 'plan.json'): M.sha(self.original / 'plan.json'),
                           str(original_module): M.sha(original_module)}))
        final = dict(physical_gpus=[4], untouched_gpus=[5, 6, 7], terminal_status='completed',
            child_exit_code=0, error=None, rlt_borrowed=True, rlt_return_dispatched=False,
            recovery_error=dict(type='OSError', error='[Errno 36] File name too long'))
        self.save(self.owner / 'final.json', final)
        self.save(self.owner / 'cleanup.json', {'all_stopped': True})
        self.release = self.owner / 'diagnostic-release.json'
        self.save(self.release, dict(cycle_id=self.stage.name, gpus=[4], terminal_status='completed',
            all_workers_stopped=True, managed_processes=[dict(pid=101, uid=20001, start=301)]))
        verified = dict(path=str(self.release), sha256=M.sha(self.release))
        self.save(self.stage / 'dojo-release-verified.json', verified)
        new_run = '/data/chenyiteng/results/rlinf-rlt/short-fixture'
        self.save(self.stage / 'plan.json', dict(group='gpu4', cycle_id=self.stage.name,
            runs={'gpu4': dict(gpus=[4], new_run=new_run, namespace='fixture-rlt')},
            repair=dict(original_stage=str(self.original), no_new_stop=True,
                        evidence={str(self.owner / 'final.json'): M.sha(self.owner / 'final.json')})))
        self.save(self.stage / 'resumed-dispatched.json', dict(cycle_id=self.stage.name,
            release=verified, runs={'gpu4': dict(run=new_run)}))
        launch = dict(identity=dict(pid=102, uid=20001, start=302), run=new_run,
            namespace='fixture-rlt', recovery_mode='resume_checkpoint')
        self.save(self.stage / 'gpu4-launched.json', launch)
        self.save(self.repair / 'repaired.json', dict(lifecycle_path=str(self.stage),
            result=dict(resumed_dispatched=True, cycle_id=self.stage.name),
            plan_sha256=M.sha(self.stage / 'plan.json'),
            return_sha256=M.sha(self.stage / 'resumed-dispatched.json'),
            original_final_preserved_sha256=M.sha(self.owner / 'final.json'),
            new_run=new_run, gpu4_launch=launch))

    def save(self, path, value):
        Path(path).write_text(json.dumps(value))

    def validate(self, helper=None):
        return M.repaired_parent_complete(self.owner, self.stage, self.module, helper)

    def test_repaired_receipt_accepts_original_failed_return(self):
        group, identity, evidence = self.validate()
        self.assertEqual(group, 'gpu4')
        self.assertEqual(identity['pid'], 100)
        self.assertEqual(evidence['owner_return'], str(self.repair / 'repaired.json'))
        self.assertFalse(M.read(self.owner / 'final.json')['rlt_return_dispatched'])

    def test_wrong_return_receipt_rejected(self):
        receipt = M.read(self.repair / 'repaired.json')
        receipt['return_sha256'] = 'bad'
        self.save(self.repair / 'repaired.json', receipt)
        with self.assertRaises(AssertionError):
            self.validate()

    def test_wrong_launch_rejected(self):
        launch = M.read(self.stage / 'gpu4-launched.json')
        launch['identity']['pid'] += 1
        self.save(self.stage / 'gpu4-launched.json', launch)
        with self.assertRaises(AssertionError):
            self.validate()

    def test_old_owner_live_rejected(self):
        with self.assertRaises(AssertionError):
            self.validate(types.SimpleNamespace(same=lambda x: x['pid'] in (100, 102)))

    def test_repaired_driver_must_be_live(self):
        with self.assertRaises(AssertionError):
            self.validate(types.SimpleNamespace(same=lambda x: False))
        self.validate(types.SimpleNamespace(same=lambda x: x['pid'] == 102))

    def test_original_final_cannot_be_rewritten(self):
        final = M.read(self.owner / 'final.json')
        final['recovery_error'] = None
        self.save(self.owner / 'final.json', final)
        with self.assertRaises(AssertionError):
            self.validate()

    def test_short_prepare_changes_only_runtime_expression(self):
        source = ("def prepare(original, stage, gpu):\n"
                  "    if True:\n"
                  "        new_run = original.with_name(original.name + '-after-' + stage.name)\n"
                  "        return new_run\n")
        module = types.SimpleNamespace(prepare=lambda: None, Path=Path)
        with patch.object(M.inspect, 'getsource', return_value=source):
            M.short_prepare(module)
        result = module.prepare(Path('/root/' + 'a' * 250), Path('/test/bell-v1-gpu4'), 4)
        self.assertEqual(str(result), '/data/chenyiteng/results/rlinf-rlt/rlt-sz3-g4-after-bell-v1-gpu4')
        self.assertLess(len(result.name.encode()), 100)

    def test_named_scope_uses_real_manifest_token_field(self):
        scope = self.repair / 'scope'
        scope.mkdir()
        runtime = scope / 'graphics_scope_runtime.py'
        runtime.write_text('# fixture runtime\n')
        manifest = scope / 'scope.json'
        activation = scope / 'activation.json'
        environment = scope / 'environment-fragment.json'
        # Actual deployed schema: manifest.token, activation.scope_id.
        self.save(manifest, dict(schema=1, token='opendwformal20261004v2',
            physical_gpus=[4, 5, 6, 7], cpu_full_mask_target=4))
        fragment = {'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST': str(manifest)}
        self.save(environment, fragment)
        self.save(activation, dict(status='active', uid=20001, manifest=str(manifest),
            manifest_sha256=M.sha(manifest), scope_id='opendwformal20261004v2',
            runtime_path=str(runtime), runtime_sha256=M.sha(runtime),
            environment_fragment_file=str(environment), environment_fragment_sha256=M.sha(environment),
            environment_fragment=fragment))
        self.save(self.stage / 'plan.json', dict(scope_manifest=str(manifest),
            scope_activation=str(activation), scope_id='opendwformal20261004v2'))
        with patch.object(M, 'NAMED_SCOPE_SHA', M.sha(runtime)), patch.object(M, 'load',
                return_value=types.SimpleNamespace(read_manifest=lambda path: M.read(path))):
            self.assertEqual(M.named_scope()['scope_id'], 'opendwformal20261004v2')


if __name__ == '__main__':
    unittest.main()
