"""Server CPU checks for ownership contract and same-budget graphics launch."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_gpu_maintenance_tests', HERE / 'maintenance_owner.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MaintenanceContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stage = Path(self.tmp.name)
        self.owner = dict(pid=10, uid=20001, boot_id='same-boot', start_ticks=20)
        self.child = dict(pid=11, uid=20001, boot_id='same-boot', start_ticks=21)
        required = [Path(m.__file__).resolve(), self.stage / 'bootstrap/sitecustomize.py', self.stage / 'scope.json',
                    m.CTRL / 'tools/reborrow_resources.py', m.SOURCE / 'tools/expo_smoke_owner.py',
                    m.SOURCE / 'tools/expo_process.py', m.SOURCE / 'tools/expo_formal_resources.py',
                    m.SOURCE / 'tools/expo_rlt_switch.py']
        self.contract = dict(version=1, purpose='expo-gpu4567-same-cycle', stage=str(self.stage),
            control=str(m.CTRL), train=str(m.TRAIN), cycle=str(m.CYCLE),
            old_current=dict(status='DRIVER_RUNNING', cycle_id=m.CYCLE.name, physical_gpus=[4, 5, 6, 7],
                             command=m.driver_command(), owner=self.owner, child=self.child),
            queue_plan_sha256={str(path / 'plan.json'): 'a' * 64 for path in m.QUEUES},
            inputs_sha256='b' * 64, cycle_plan_sha256='c' * 64,
            files_sha256={str(path): 'd' * 64 for path in required}, graphics_probe_verified=True)

    def test_exact_same_cycle_contract_is_accepted(self):
        m.validate_contract(self.contract, self.stage)

    def test_other_uid_boot_or_gpu_set_is_rejected(self):
        for kind in ('uid', 'boot', 'gpus'):
            bad = copy.deepcopy(self.contract)
            if kind == 'uid':
                bad['old_current']['child']['uid'] = 123
            elif kind == 'boot':
                bad['old_current']['child']['boot_id'] = 'different-boot'
            else:
                bad['old_current']['physical_gpus'] = [0, 1, 2, 3]
            with self.assertRaises(ValueError):
                m.validate_contract(bad, self.stage)

    def test_wrong_training_budget_is_rejected(self):
        bad = copy.deepcopy(self.contract)
        index = bad['old_current']['command'].index('--max-physical-actions') + 1
        bad['old_current']['command'][index] = '30000'
        with self.assertRaises(ValueError):
            m.validate_contract(bad, self.stage)

    def test_missing_scope_pin_is_rejected(self):
        del self.contract['files_sha256'][str(self.stage / 'scope.json')]
        with self.assertRaisesRegex(ValueError, 'source pin'):
            m.validate_contract(self.contract, self.stage)

    def test_graphics_probe_required_before_adoption(self):
        self.contract['graphics_probe_verified'] = False
        with self.assertRaisesRegex(ValueError, 'Graphics probe'):
            m.validate_contract(self.contract, self.stage)

    def test_environment_scopes_parent_and_spawn_before_gpu_imports(self):
        with patch.dict(os.environ, {'DISPLAY': ':0', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '/unrelated'}):
            env = m.child_environment(self.stage, 'test-scope')
        self.assertNotIn('DISPLAY', env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], ','.join(m.UUIDS))
        self.assertEqual(env['RLINF_EXPO_GPU_SCOPE_MANIFEST'], str(self.stage / 'scope.json'))
        self.assertEqual(env['__GL_APPLICATION_PROFILE'], '1')
        self.assertEqual(env['PYTHONPATH'].split(':')[0], str(self.stage / 'bootstrap'))
        self.assertEqual(env['EXPO_SMOKE_OWNER_SCOPE'], 'test-scope')

    def test_resume_keeps_original_training_and_checkpoint_paths(self):
        command = m.driver_command()
        self.assertEqual(command[command.index('--inputs') + 1], str(m.TRAIN / 'inputs.json'))
        self.assertEqual(command[command.index('--run') + 1], str(m.TRAIN / 'run'))
        self.assertEqual(command[command.index('--resume') + 1], str(m.TRAIN / 'run/checkpoint-latest.pt'))
        self.assertEqual(command[command.index('--max-physical-actions') + 1], '20000')

    def test_request_requires_exact_owner_and_contract(self):
        request = dict(action='resume', contract_sha256='a' * 64, owner=self.owner)
        self.assertTrue(m.request_valid(request, 'a' * 64, self.owner))
        self.assertFalse(m.request_valid(request, 'b' * 64, self.owner))
        self.assertFalse(m.request_valid(request, 'a' * 64, dict(self.owner, start_ticks=21)))
        self.assertFalse(m.request_valid(dict(request, action='launch-other'), 'a' * 64, self.owner))

    def test_graphics_only_context_is_included_in_nvml_parse(self):
        xml = '<nvidia_smi_log><gpu><uuid>physical0</uuid><processes><process_info><pid>11</pid>' \
              '<type>G</type><used_memory>7 MiB</used_memory></process_info></processes></gpu></nvidia_smi_log>'
        with patch.object(m.subprocess, 'check_output', return_value=xml):
            self.assertEqual(m.all_gpu_processes(), [dict(index=0, gpu='physical0', pid=11, type='G', memory='7 MiB')])

    def test_managed_gpu0_teardown_lag_is_waited_out(self):
        clock = [0.0]
        def rows():
            return [dict(gpu='physical0', pid=11)] if clock[0] < 3 else []
        with patch.object(m, 'all_gpu_processes', side_effect=rows), \
             patch.object(m.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(m.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            m.wait_release([self.child], lambda row: False, timeout=10)
        self.assertGreaterEqual(clock[0], 5)

    def test_foreign_gpu0_does_not_block_or_enter_cleanup_scope(self):
        clock = [0.0]
        with patch.object(m, 'all_gpu_processes', return_value=[dict(gpu='physical0', pid=999)]), \
             patch.object(m.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(m.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            m.wait_release([self.child], lambda row: False, timeout=10)
        self.assertEqual(clock[0], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
