import json
from pathlib import Path
import tempfile
import threading
import unittest
import run_queue as q


class QueueTests(unittest.TestCase):
    def test_native_fatal_codes_and_text(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'child.log';p.write_text('ordinary startup')
            for code in (99,134,139,-6,-11):self.assertTrue(q.child_fatal(code,p))
            self.assertFalse(q.child_fatal(1,p))
            for message in ('CUDA error: bad','Illegal memory access','device-side assert','Invalid PhysX transform'):
                p.write_text(message);self.assertTrue(q.child_fatal(0,p))

    def test_config_only_gpu_may_differ(self):
        self.assertTrue(q._same_config({'gpu':7,'H':50},{'gpu':4,'H':50}))
        self.assertFalse(q._same_config({'gpu':7,'H':50},{'gpu':4,'H':25}))

    def test_cancel_sets_both_events(self):
        obj=object.__new__(q.Queue);obj.stop=threading.Event();obj.cancelled=threading.Event()
        obj.cancel();self.assertTrue(obj.stop.is_set());self.assertTrue(obj.cancelled.is_set())

    def test_fingerprint_tracks_wrapper_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);repo=base/'repo';out=base/'out';out.mkdir()
            module=repo/'tools/pi05_signals';module.mkdir(parents=True)
            wrapper=repo/q.WRAPPER;wrapper.parent.mkdir(parents=True);wrapper.write_text('original')
            for name in q.CORE_FILES:(module/name).write_text('original')
            config=out/'task.json';config.write_text('{}')
            (out/'environment.json').write_text('{}')
            (out/'manifest.json').write_text(json.dumps({'configs':[str(config)],'smoke':[str(config)]}))
            first=q.signal_fingerprint(out,repo)
            wrapper.write_text('modified');self.assertNotEqual(first,q.signal_fingerprint(out,repo))
            wrapper.write_text('original');self.assertEqual(first,q.signal_fingerprint(out,repo))
            config.write_text('{"H":25}');self.assertNotEqual(first,q.signal_fingerprint(out,repo))

    def test_duplicate_claim_stems_rejected(self):
        with self.assertRaises(RuntimeError):q.manifest_configs({'configs':['/a/x.json','/b/x.json'],'smoke':['/a/x.json']})


if __name__=='__main__':unittest.main()
