"""CPU-only cache behavior and sampling equivalence, run on the experiment server."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch

module_path = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else Path(__file__).parents[1] / 'rlinf/algorithms/expo_ft/formal_replay.py'
spec = importlib.util.spec_from_file_location('cache_replay', module_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.set_num_threads(2)


class ReplayCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.demo = root / 'demo'
        (self.demo / 'data/chunk-000').mkdir(parents=True)
        (self.demo / 'meta').mkdir()
        (self.demo / 'prepared.json').write_text(json.dumps(dict(complete=True, dataset_revision='cache-fixture', episodes=2, frames=120)))
        (self.demo / 'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0, task='fixture')) + '\n')
        for ep in range(2):
            image = io.BytesIO()
            Image.fromarray(np.full((8, 8, 3), 30 + ep, dtype=np.uint8)).save(image, format='PNG')
            data = {key: [{'bytes': image.getvalue()}] * 60 for key in m.IMAGE_COLUMNS}
            data.update({'action': [[0.01 * ep] * 14] * 60, 'observation.state': [[0.] * 14] * 60, 'task_index': [0] * 60})
            pq.write_table(pa.table(data), self.demo / f'data/chunk-000/episode_{ep:06d}.parquet')
        self.pool = root / 'pool'
        replay = self.replay()
        obs = dict(main_images=torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
                   wrist_images=torch.zeros(1, 2, 8, 8, 3, dtype=torch.uint8),
                   states=torch.zeros(1, 14), task_descriptions=['fixture'])
        replay.append_episode([obs] * 60, torch.zeros(60, 14), [0.] * 59 + [1.], True, False, True, 'online-000000', obs)

    def tearDown(self):
        self.tmp.cleanup()

    def replay(self, limit=64 * 1024**3):
        return m.FormalReplay(self.pool, self.demo, seed=42, cache_limit_bytes=limit)

    def test_online_and_demo_hits_do_not_reload_or_rehash(self):
        r = self.replay()
        for entry in [r.demo_entries[0], r.online_entries[0]]:
            value = r._cached(entry)
            original = m._digest
            def forbidden(*_):
                raise AssertionError('Rehashed cached payload')
            m._digest = forbidden
            try:
                self.assertIs(value, r._cached(entry))
            finally:
                m._digest = original
        self.assertEqual(r.cache_hits, 2)
        self.assertEqual(r.cache_misses, 2)

    def test_q_fm_samples_and_resume_state_match_without_cache(self):
        cold, warm = self.replay(0), self.replay()
        for _ in range(2):
            left, right = cold._draw(12), warm._draw(12)
            self.assertEqual(left, right)
            self.assertEqual(m._digest(cold._windows(left, m.C)), m._digest(warm._windows(right, m.C)))
            self.assertEqual(m._digest(cold.sample_fm(8)), m._digest(warm.sample_fm(8)))
        self.assertEqual(m._digest(cold.state_dict()), m._digest(warm.state_dict()))
        before = m._digest(warm.state_dict())
        warm.load_state_dict(warm.state_dict())
        self.assertEqual(before, m._digest(warm.state_dict()))
        self.assertEqual(warm.cache_bytes, 0)
        self.assertEqual(len(warm._cache), 0)

    def test_byte_limit_lru_and_oversized_bypass(self):
        r = self.replay()
        a, b = r.demo_entries
        va, vb = r._cached(a), r._cached(b)
        limit = max(m._resident_bytes(va), m._resident_bytes(vb))
        limited = self.replay(limit)
        limited._cached(a); limited._cached(b)
        self.assertLessEqual(limited.cache_bytes, limit)
        self.assertEqual(list(limited._cache), [(b['kind'], b['id'])])
        self.assertEqual(m._digest(limited._cached(a)), m._digest(va))
        zero = self.replay(0)
        self.assertEqual(m._digest(zero._cached(a)), m._digest(va))
        self.assertEqual(zero.cache_bytes, 0)

    def test_changed_cached_file_is_rejected(self):
        r = self.replay()
        entry = r.online_entries[0]
        r._cached(entry)
        path = r.root / entry['path']
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000000))
        with self.assertRaises(RuntimeError):
            r._cached(entry)

    def test_tensor_storage_alias_not_double_counted(self):
        value = torch.zeros(1024, dtype=torch.uint8)
        one = m._resident_bytes([value])
        aliases = m._resident_bytes([value, value, value[:]])
        self.assertLess(aliases - one, 1024)


if __name__ == '__main__':
    unittest.main()
