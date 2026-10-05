"""CPU tests for label integrity, first success and unknown-output accounting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from native_binary_recorder import NativeBinaryRecorder
from rynn_binary_probe import summarize
from prepare_bell_reward_samples import select_candidates
from run_native_eval import owner_markers


class BinaryIntegrityTests(unittest.TestCase):
    def test_owner_markers_are_required_as_a_pair(self):
        with self.assertRaises(ValueError):
            owner_markers({"OPENDW_SMOKE_OWNER_TOKEN": "token"})
        markers = {"OPENDW_SMOKE_OWNER_TOKEN": "token", "OPENDW_SMOKE_OWNER_PHASE": "precheck"}
        self.assertEqual(owner_markers(markers), markers)

    def test_reward_selection_excludes_latched_from_primary_binary_labels(self):
        records = [dict(episode_uid="a", native_success=[None, False, True, True],
            action_steps=[0, 32, 64, 96], first_success_position=2, record_path="a.json", instruction="Click bell."),
            dict(episode_uid="b", native_success=[None, False, False, False], action_steps=[0, 32, 64, 96],
                first_success_position=None, record_path="b.json", instruction="Click bell.")]
        selected, available = select_candidates(records)
        self.assertEqual(available, dict(first_success=1, near_false=2, post_success_latched=1))
        self.assertEqual([r["label"] for r in selected], [1, 0, 0, -1])
        self.assertEqual([r["frame_index"] for r in selected], [2, 1, 3, 3])
        self.assertTrue(selected[-1]["simulator_success"])

    def test_success_first_hit_and_failure_only_full_horizon(self):
        env = SimpleNamespace(num_envs=2, task_name="adjust_bottle", elapsed_steps=np.zeros(2, dtype=int),
                              cfg=SimpleNamespace(max_episode_steps=64))
        obs = dict(main_images=np.zeros((2, 16, 20, 3), dtype=np.uint8),
                   task_descriptions=["Adjust the bottle."] * 2)
        with tempfile.TemporaryDirectory() as root:
            recorder = NativeBinaryRecorder(env, root)
            recorder.reset(obs, None, [9, 10])
            env.elapsed_steps[:] = 32
            obs["main_images"][:] = 32
            recorder.observe(obs, {"success": np.asarray([True, False])})
            paths = list(Path(root).glob("*.json"))
            self.assertEqual(len(paths), 1)
            first = json.loads(paths[0].read_text())
            self.assertTrue(first["reference_success"])
            self.assertEqual(first["terminal_action_steps"], 32)
            env.elapsed_steps[:] = 64
            obs["main_images"][:] = 64
            recorder.observe(obs, {"success": np.asarray([False, False])})
            rows = [json.loads(p.read_text()) for p in Path(root).glob("*.json")]
            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(r["reference_success"] for r in rows), [False, True])
            # Successful record remains from32; continued motion did not relabel it.
            self.assertEqual(next(r for r in rows if r["reference_success"])["terminal_action_steps"], 32)
            failure = next(r for r in rows if not r["reference_success"])
            self.assertEqual(failure["terminal_action_steps"], 64)

    def test_early_reset_is_not_failure(self):
        env = SimpleNamespace(num_envs=1, task_name="adjust_bottle", elapsed_steps=np.zeros(1),
                              cfg=SimpleNamespace(max_episode_steps=384))
        obs = dict(main_images=np.zeros((1, 16, 20, 3), dtype=np.uint8), task_descriptions=["Adjust bottle."])
        with tempfile.TemporaryDirectory() as root:
            recorder = NativeBinaryRecorder(env, root)
            recorder.reset(obs, None, [0])
            env.elapsed_steps[:] = 32
            recorder.observe(obs, {"success": np.asarray([False])})
            with self.assertRaisesRegex(RuntimeError, "before a verified"):
                recorder.reset(obs, None, [1])
            self.assertEqual(list(Path(root).glob("*.json")), [])

    def test_unknown_excluded_and_auc_handles_overlap(self):
        row = lambda label, value, answer, status="ok": dict(reference_success=label,
            last_remaining_value=value, language=dict(success=answer, parse_status=status))
        result = summarize([row(True, .2, False), row(True, .8, None, "missing_success"),
                            row(False, .4, True), row(False, .9, False)])
        self.assertEqual(result["language"], dict(true_positive=0, false_negative=1,
            false_positive=1, true_negative=1, unknown=1))
        self.assertEqual(result["auc_lower_value_predicts_success"], .75)
        self.assertFalse(result["strict_low_value_separation_on_this_sample"])
        self.assertIsNone(result["deployed_threshold"])

    def test_reward_native_keeps_post_success_and_unknown_reset(self):
        env = SimpleNamespace(num_envs=1, task_name="click_bell", elapsed_steps=np.zeros(1, dtype=int),
                              cfg=SimpleNamespace(max_episode_steps=64))
        obs = dict(main_images=np.zeros((1, 16, 20, 3), dtype=np.uint8), task_descriptions=["Click bell."])
        with tempfile.TemporaryDirectory() as root, patch.dict("os.environ", {"RYNN_BINARY_CAPTURE_MODE": "reward_native"}):
            recorder = NativeBinaryRecorder(env, root)
            recorder.reset(obs, None, [0])
            env.elapsed_steps[:] = 32
            recorder.observe(obs, {"success": np.asarray([True])})
            self.assertEqual(list(Path(root).glob("*.json")), [])
            env.elapsed_steps[:] = 64
            recorder.observe(obs, {"success": np.asarray([False])})
            path = next(Path(root).glob("*.json"))
            row = json.loads(path.read_text())
            self.assertEqual(row["native_success"], [None, True, False])
            self.assertTrue(row["reference_success"])
            self.assertFalse(row["native_success_at_end"])
            self.assertEqual(row["terminal_action_steps"], 32)
            self.assertEqual(row["collection_action_steps"], 64)
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                self.assertEqual(data["native_frames"].shape, (3, 16, 20, 3))


if __name__ == "__main__":
    unittest.main()
