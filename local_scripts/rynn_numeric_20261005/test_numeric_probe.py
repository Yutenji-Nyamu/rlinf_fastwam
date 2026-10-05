"""Server CPU checks for extraction/report semantics; no model/GPU loads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("probe", Path(__file__).with_name("rynn_numeric_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class NumericProbeTests(unittest.TestCase):
    def test_spearman_and_ties(self):
        self.assertAlmostEqual(probe.spearman([0, 1, 2], [3, 2, 1]), -1)
        self.assertIsNone(probe.spearman([0, 1, 2], [2, 2, 2]))
        self.assertEqual(probe.ranks([3, 1, 1]).tolist(), [2, .5, .5])

    def test_trend_direction_and_temporal_controls(self):
        rows = [dict(id=str(i), episode_uid="e0", kind="expert_prefix", prefix_fraction=i/4,
                     last_remaining_value=10-i*2) for i in range(5)]
        rows += [dict(episode_uid="e0", kind="repeat_final", last_remaining_value=3),
                 dict(episode_uid="e0", kind="reversed", last_remaining_value=9),
                 dict(episode_uid="e0", kind="return_to_initial", last_remaining_value=11)]
        result = probe.summarize(rows)
        self.assertEqual(result["endpoints_lower_count"], 1)
        self.assertEqual(result["median_initial_minus_final"], 8)
        self.assertEqual(result["expert_trends"][0]["reversed_minus_forward"], 7)
        self.assertEqual(result["expert_trends"][0]["return_minus_initial"], 1)
        self.assertEqual(result["expert_trends"][0]["full_prefix_minus_repeated_final"], -1)
        self.assertIsNone(result["native_eval"]["auc_low_value_predicts_success"])

    def test_native_auc_excludes_synthetic_and_prefixes(self):
        rows = [dict(episode_uid=str(i), kind="native_eval", prefix_fraction=1,
                     reference_success=ok, last_remaining_value=value)
                for i, (ok, value) in enumerate([(True, 1), (True, 2), (False, 2), (False, 4)])]
        rows += [dict(episode_uid="synthetic", kind="reversed", reference_success=False, last_remaining_value=-100),
                 dict(episode_uid="prefix", kind="native_eval", reference_success=False,
                      prefix_fraction=.5, last_remaining_value=-100)]
        summary = probe.summarize(rows)["native_eval"]
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failure_count"], 2)
        self.assertEqual(summary["auc_low_value_predicts_success"], .875)

    def test_frame_manifest_and_synthetic_label_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            frames = np.zeros((8,256,320,3), dtype=np.uint8)
            np.savez(path/"samples.npz", test=frames)
            case = dict(id="e0/reversed", frames_key="test", episode_uid="e0", kind="reversed",
                instruction="Lift the bottle.", source="h5", label_origin="synthetic reversal",
                array_sha256=probe.sha(frames.tobytes()), frame_indices=list(range(7,-1,-1)), reference_success=None)
            def write():
                (path/"cases.json").write_text(json.dumps(dict(schema_version=1,cases=[case])), encoding="utf-8")
            write()
            cases, arrays = probe.load_cases(path/"cases.json", path/"samples.npz")
            self.assertEqual(len(cases), 1)
            self.assertEqual(arrays["test"].shape[0], 8)
            case["reference_success"] = False
            write()
            with self.assertRaisesRegex(ValueError, "Synthetic temporal"):
                probe.load_cases(path/"cases.json", path/"samples.npz")
            case["reference_success"] = None
            case["array_sha256"] = "0"*64
            write()
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                probe.load_cases(path/"cases.json", path/"samples.npz")

    def test_official_source_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"inference.py"
            source.write_text("def main():\n    def run_batch(samples):\n        return []\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pinned revision"):
                probe.official_numeric(source)

    def test_head_capture_preserves_batch_and_head_axes(self):
        import torch
        from types import SimpleNamespace
        class FakeModel:
            config = SimpleNamespace(num_value_heads=2)
            def __call__(self, **kwargs):
                return SimpleNamespace(value=SimpleNamespace(pred_value=torch.arange(32).reshape(2,16)),
                    relative=SimpleNamespace(pred_value=torch.arange(14)))
        wrapper = probe.CaptureModel(FakeModel())
        wrapper(input_ids=torch.zeros((2,4)))
        self.assertEqual(wrapper.captured["absolute_slots"][0], list(range(8,16)))
        self.assertEqual(wrapper.captured["absolute_slots"][1], list(range(16,24)))
        self.assertEqual(wrapper.captured["relative_slots"][1], list(range(7,14)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
