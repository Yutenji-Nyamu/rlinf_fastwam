"""Focused CPU checks; run on SZ2 before launching the resource owner."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import expo_formal_owner as owner
import expo_formal_resources as resources


class OwnerSafety(unittest.TestCase):
    def test_recipe_allows_only_source_manifest_changes(self):
        original = {'formal': {'max_physical_actions': 20000}, 'port_source_manifest': {'x': 'old'}}
        updated = {'formal': {'max_physical_actions': 20000}, 'port_source_manifest': {'x': 'new'}}
        self.assertEqual(owner.recipe(original), owner.recipe(updated))
        updated['formal']['max_physical_actions'] = 10
        self.assertNotEqual(owner.recipe(original), owner.recipe(updated))

    def test_pid_reuse_after_open_does_not_signal(self):
        row = {'uid': 20001, 'pid': 123, 'start': 456}
        helper = types.SimpleNamespace(process_tree=lambda roots: {123: row}, same=Mock(side_effect=[True, False]))
        resources.pin_signals(helper)
        helper.process_tree([123])
        with patch.object(resources, 'pidfd_open', return_value=42), \
                patch.object(resources, 'pidfd_send') as send, patch.object(resources.os, 'close') as close:
            helper.os.kill(123, signal.SIGTERM)
        send.assert_not_called()
        close.assert_called_once_with(42)

    def test_unregistered_pid_never_opens_pidfd(self):
        helper = types.SimpleNamespace(process_tree=lambda roots: {}, same=Mock())
        resources.pin_signals(helper)
        with patch.object(resources, 'pidfd_open') as opened:
            with self.assertRaises(AssertionError):
                helper.os.kill(123, signal.SIGKILL)
        opened.assert_not_called()

    def test_valid_registered_identity_signals_through_fd(self):
        row = {'uid': 20001, 'pid': 123, 'start': 456}
        helper = types.SimpleNamespace(process_tree=lambda roots: {123: row}, same=Mock(return_value=True))
        resources.pin_signals(helper)
        helper.process_tree([123])
        with patch.object(resources, 'pidfd_open', return_value=42), \
                patch.object(resources, 'pidfd_send') as send, patch.object(resources.os, 'close'):
            helper.os.kill(123, signal.SIGTERM)
        send.assert_called_once_with(42, signal.SIGTERM)

    def test_native_receipt_cannot_be_reused_after_inputs_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            (source / 'tests').mkdir(parents=True)
            test = source / 'tests/native_expo_lifecycle.py'
            test.write_text('pass\n')
            (root / 'inputs.json').write_text('{"version": 1}')
            receipt = dict(ok=True, mode='fixed', inputs_sha256=owner.sha(root / 'inputs.json'), test_sha256=owner.sha(test))
            (root / 'native-check.json').write_text(json.dumps(receipt))
            with patch.object(owner, 'ROOT', root), patch.object(owner, 'SOURCE', source):
                self.assertTrue(owner.native_receipt()['ok'])
                (root / 'inputs.json').write_text('{"version": 2}')
                with self.assertRaises(AssertionError):
                    owner.native_receipt()

    def test_diagnostic_cannot_run_script_outside_source_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            (source / 'tests').mkdir(parents=True)
            script = source / 'outside.py'
            script.write_text('pass\n')
            config = dict(argv=[owner.EXPO_PY, '-X', 'faulthandler', '-u', '-B', str(script)],
                source_sha256={'outside.py': owner.sha(script)}, timeout_seconds=900)
            (root / 'diagnostic.json').write_text(json.dumps(config))
            with patch.object(owner, 'ROOT', root), patch.object(owner, 'SOURCE', source):
                with self.assertRaises(AssertionError):
                    owner.diagnostic_command(root / 'diagnostic.json')

    def test_diagnostic_accepts_symlinked_source_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / 'nvme/source'
            (actual / 'tests').mkdir(parents=True)
            source = root / 'source'
            source.symlink_to(actual, target_is_directory=True)
            script = source / 'tests/diagnostic.py'
            script.write_text('pass\n')
            command = [owner.EXPO_PY, '-X', 'faulthandler', '-u', '-B', str(script)]
            config = dict(argv=command, source_sha256={'tests/diagnostic.py': owner.sha(script)}, timeout_seconds=900)
            (root / 'diagnostic.json').write_text(json.dumps(config))
            with patch.object(owner, 'ROOT', root), patch.object(owner, 'SOURCE', source):
                actual_command, timeout = owner.diagnostic_command(root / 'diagnostic.json')
            self.assertEqual(actual_command, command)
            self.assertEqual(timeout, 900)


if __name__ == '__main__':
    unittest.main()
