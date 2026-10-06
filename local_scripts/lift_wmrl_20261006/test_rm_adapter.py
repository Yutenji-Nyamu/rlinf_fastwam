"""Server CPU checks for the new model binding and preserved pixel interface."""
import tempfile
from pathlib import Path
import unittest

import torch
from rm_inference import PREPROCESS, SingleTaskReward, sha256
from rm_adapter import LiftPotReward
from patch_lift import service_source


class RewardAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "test.pt"
        cls.original = SingleTaskReward().float().eval().requires_grad_(False)
        cls.payload = dict(schema_version=1, architecture="resnet18_mlp256", preprocess=PREPROCESS,
                           task_name="lift_pot", task_config="demo_clean",
                           model_state_dict=cls.original.state_dict())
        torch.save(cls.payload, cls.path)
        cls.hash = sha256(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_scores_equal_training_reference_for_dict_and_both_pixel_layouts(self):
        model = LiftPotReward(self.path, expected_sha256=self.hash)
        pixels = torch.randint(256, (2, 256, 320, 3), dtype=torch.uint8)
        expected = self.original.compute_reward(pixels)
        for images in (pixels, pixels.permute(0, 3, 1, 2), {"main_images": pixels}):
            scores = model.compute_reward(images, ["Lift the pot"] * 2)
            self.assertEqual(tuple(scores.shape), (2,))
            self.assertTrue(torch.isfinite(scores).all())
            self.assertTrue(((0 <= scores) & (scores <= 1)).all())
            torch.testing.assert_close(scores, expected, rtol=0, atol=0)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_wrong_checkpoint_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            LiftPotReward(self.path, expected_sha256="0" * 64)

    def test_wrong_task_rejected(self):
        path = Path(self.temp.name) / "wrong-task.pt"
        torch.save(dict(self.payload, task_name="click_bell"), path)
        with self.assertRaisesRegex(ValueError, "lift_pot/demo_clean"):
            LiftPotReward(path, expected_sha256=sha256(path))

    def test_plain_legacy_state_dictionary_rejected(self):
        path = Path(self.temp.name) / "legacy.pt"
        torch.save(self.original.state_dict(), path)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            LiftPotReward(path, expected_sha256=sha256(path))

    def test_patch_rejects_unknown_or_already_patched_donor(self):
        with self.assertRaisesRegex(ValueError, "Donor source differs"):
            service_source("from rm_adapter import LiftPotReward\n")


if __name__ == "__main__":
    unittest.main()
