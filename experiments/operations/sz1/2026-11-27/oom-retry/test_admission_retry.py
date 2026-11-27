import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch,Mock
import queue_admission as admission


class PairAdmissionTest(unittest.TestCase):
    def test_normal_launcher(self):
        with TemporaryDirectory() as root, patch.object(admission, "same", side_effect=AssertionError):
            admission.wait_for_oom_retry(Path(root))

    def test_each_successor_waits_for_same_gate(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            gate = root / "gate-release.json"
            primary = root / "identity.json"
            policy = {"oom_retry_gate":{"identity":{"pid":33}, "release":str(gate), "primary_identity":str(primary)}}
            for card in ("4", "7"):
                gate.unlink(missing_ok=True)
                d=root/card;d.mkdir();(d/"wait.json").touch()
                with patch.object(admission,"read",return_value=policy), patch.object(admission,"same",return_value=True), patch.object(admission.time,"sleep",side_effect=lambda _:gate.touch()) as pause:
                    admission.wait_for_oom_retry(d)
                    self.assertEqual(pause.call_count,1)

    def test_dead_gate_with_live_relaunched_job(self):
        with TemporaryDirectory() as root:
            d=Path(root);(d/"wait.json").touch()
            p=d/"primary.json";p.touch();g=d/"released.json"
            guards={"oom_retry_gate":{"identity":{"pid":1},"release":str(g),"primary_identity":str(p)}}
            def get(path):return guards if path.name=="wait.json" else {"pid":2}
            live=iter((False,True,False,False))
            with patch.object(admission,"read",side_effect=get),patch.object(admission,"same",side_effect=lambda obj:next(live)),patch.object(admission.time,"sleep") as delay:
                admission.wait_for_oom_retry(d);self.assertEqual(delay.call_count,1)

