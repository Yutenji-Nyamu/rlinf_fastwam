"""CPU-only profile representation checks; no GPU/Ray imports or activation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('graphics_scope_runtime.py')
SPEC = importlib.util.spec_from_file_location('numeric_named_scope', SOURCE)
SCOPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCOPE)


class NamedProfileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = dict(schema=1, uid=os.getuid(), hostname=socket.gethostname(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            physical_gpus=[4, 5, 6, 7], cpu_full_mask_target=4,
            existing_profiles={}, profile_search_paths=[], cards={})
        for gpu in (4, 5, 6, 7):
            self.manifest['cards'][str(gpu)] = dict(physical_gpu=gpu, gpu_uuid='GPU-unit-' + str(gpu),
                minor=gpu, mask=1 << gpu, commname='odwf' + str(gpu) + '-unit')
        for key in ('marker', 'bootstrap', 'runtime'):
            path = self.root / key
            path.write_text('unit-fixture')
            path.chmod(0o600)
            self.manifest[key + '_path'] = str(path)
            self.manifest[key + '_sha256'] = SCOPE.digest(path)
        self.rules = [dict(pattern=dict(feature='commname', matches=self.manifest['cards'][str(g)]['commname']),
            profile=dict(name='chenyiteng-' + self.manifest['cards'][str(g)]['commname'],
                         settings=['EGLVisibleDGPUDevices', 1 << g])) for g in (4, 5, 6, 7)]

    def read(self, rules):
        profile = self.root / 'profile.json'
        profile.write_text(json.dumps(dict(rules=rules)))
        profile.chmod(0o600)
        self.manifest.update(profile_path=str(profile), profile_sha256=SCOPE.digest(profile))
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(self.manifest))
        path.chmod(0o600)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='4'):
            return SCOPE.read_manifest(path)

    def test_named_and_original_forms_keep_gpu4(self):
        self.assertEqual(self.read(copy.deepcopy(self.rules))['physical_gpu'], 4)
        plain = copy.deepcopy(self.rules)
        for rule in plain:
            rule['profile'] = rule['profile']['settings']
        self.assertEqual(self.read(plain)['mask'], 16)

    def test_wrong_mask_rejected(self):
        self.rules[0]['profile']['settings'][1] = 1
        with self.assertRaisesRegex(RuntimeError, 'four explicit card rules'):
            self.read(self.rules)

    def test_wrong_name_rejected(self):
        self.rules[0]['profile']['name'] = 'other-name'
        with self.assertRaisesRegex(RuntimeError, 'pinned commname'):
            self.read(self.rules)

    def test_extra_profile_key_rejected(self):
        self.rules[0]['profile']['unexpected'] = True
        with self.assertRaisesRegex(RuntimeError, 'pinned commname'):
            self.read(self.rules)


if __name__ == '__main__':
    unittest.main()
