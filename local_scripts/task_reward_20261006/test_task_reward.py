"""Server CPU checks for label alignment, split isolation and checkpoint contract."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from prepare_dataset import assign_splits, load_aligned_images, valid_indices
from rm_inference import PREPROCESS, SingleTaskReward
from rm_train import choose_threshold, metrics


class DatasetTests(unittest.TestCase):
    def test_reset_and_latched_success_are_not_training_labels(self):
        row = dict(native_success=[None, False, True, True], action_steps=[0, 32, 64, 96],
                   first_success_position=2, reference_success=True, collection_action_steps=96, max_episode_steps=96)
        self.assertEqual(valid_indices(row), [1, 2])
        row.update(native_success=[None, False, False, False], first_success_position=None, reference_success=False, collection_action_steps=64)
        with self.assertRaises(ValueError):
            valid_indices(row)

    def test_k8_duplicate_indices_do_not_duplicate_labels(self):
        indices = [0, 0, 1, 1, 2, 2, 3, 3]
        frames = np.stack([np.full((8, 8, 3), index, np.uint8) for index in indices])
        row = dict(capture_mode="binary_terminal", frame_indices=indices, native_success=[None, False, False, True],
                   array_sha256=hashlib.sha256(frames.tobytes()).hexdigest())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episode.json"
            np.savez(path.with_suffix(".npz"), frames=frames)
            _, mapping, _ = load_aligned_images(row, path)
            self.assertEqual(mapping, {0: 0, 1: 2, 2: 4, 3: 6})
            frames[1] += 1
            np.savez(path.with_suffix(".npz"), frames=frames)
            row["array_sha256"] = hashlib.sha256(frames.tobytes()).hexdigest()
            with self.assertRaises(ValueError):
                load_aligned_images(row, path)

    def test_repeated_reset_seeds_stay_grouped(self):
        records = [dict(task_name="task", seed=seed, reference_success=seed % 2 == 0) for seed in range(20)]
        records.append(dict(task_name="task", seed=0, reference_success=False))
        plan = assign_splits(records)
        self.assertEqual(len(plan), 20)
        self.assertEqual(set(plan.values()), {"train", "val", "test"})
        self.assertEqual(plan, assign_splits(list(reversed(records))))

    def test_threshold_rejects_early_false_completion(self):
        labels, scores, episodes = [0, 1, 0, 1], [.8, .9, .2, .7], ["a", "a", "b", "b"]
        chosen = choose_threshold(labels, scores, episodes, 0.)
        self.assertEqual((chosen["tp"], chosen["fp"], chosen["fn"]), (1, 0, 1))
        self.assertEqual(metrics(labels, scores, episodes, .5)["false_alarm_episodes"], 1)


class ModelTests(unittest.TestCase):
    def test_strict_checkpoint_and_deployed_score_interface(self):
        torch.set_num_threads(1)
        model = SingleTaskReward().eval()
        images = torch.randint(0, 256, (2, 64, 80, 3), dtype=torch.uint8)
        expected = model.compute_reward(images, ["ignored"] * 2)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rm.pt"
            payload = dict(schema_version=1, architecture="resnet18_mlp256", preprocess=PREPROCESS,
                           task_name="task", model_state_dict=model.state_dict())
            torch.save(payload, path)
            restored = SingleTaskReward(checkpoint_path=path)
            actual = restored.compute_reward(images, ["different"] * 2)
            torch.testing.assert_close(expected, actual)
            self.assertEqual(tuple(actual.shape), (2,))
            self.assertTrue(bool(((actual >= 0) & (actual <= 1)).all()))
            torch.testing.assert_close(actual, restored.compute_reward(images.float() / 255, None))
            payload["model_state_dict"].pop("backbone.fc.3.weight")
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                SingleTaskReward(checkpoint_path=path)


if __name__ == "__main__":
    unittest.main()
