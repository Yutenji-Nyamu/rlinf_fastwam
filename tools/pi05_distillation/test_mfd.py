# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Analytic MFD target/gradient checks; run on server CPU without CUDA context."""

import unittest
from unittest.mock import patch

import torch

from rlinf.algorithms.pi05_distillation import mfd
from tools.pi05_distillation.prepare_mfd import config_from_dmd


class AffineVelocity(torch.nn.Module):
    def __init__(self, slope, bias=0.0):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(float(slope)))
        self.bias = bias

    def forward(self, x, t, context):
        return self.slope * x + self.bias


class MFDTests(unittest.TestCase):
    def test_interval_distribution_and_conditional_bounds(self):
        probabilities = mfd.interval_probabilities(0.02)
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0).double())
        torch.testing.assert_close(
            (probabilities * torch.arange(1, 51)).sum(), torch.tensor(17.17).double()
        )
        torch.manual_seed(19)
        for k in (1, 17, 50):
            with patch.object(torch, "multinomial", return_value=torch.tensor([k - 1])):
                start, end, steps = mfd.sample_intervals(5000, "cpu", step_size=0.02)
            self.assertEqual(steps, k)
            self.assertTrue(((end >= 0) & (start <= 1) & (end <= start)).all())
            duration = start - end
            self.assertGreaterEqual(float(duration.min()), (k - 1) * 0.02 - 1e-6)
            self.assertLessEqual(float(duration.max()), k * 0.02 + 1e-6)
            # Conditional r is uniform on [0, 1-duration].
            ratio = end / (1 - duration).clamp_min(1e-7)
            self.assertAlmostEqual(float(ratio.mean()), 0.5, delta=0.02)
        with self.assertRaises(ValueError):
            mfd.interval_probabilities(0)

    def test_batched_integral_matches_scalar_euler_and_follows_own_path(self):
        x = torch.tensor([[[2.0]], [[-1.0]], [[0.4]]], requires_grad=True)
        start = torch.tensor([0.9, 0.7, 0.3])
        end = torch.tensor([0.4, 0.1, 0.3])
        model = AffineVelocity(0.6, 0.2)
        result = mfd.average_velocity(model, x, start, end, None, steps=5)
        scalar = []
        for i in range(3):
            value, delta = float(x[i].detach()), float((start[i] - end[i]) / 5)
            velocities = []
            for _ in range(5):
                velocity = 0.6 * value + 0.2
                velocities.append(velocity)
                value -= delta * velocity
            scalar.append(sum(velocities) / 5)
        torch.testing.assert_close(result.flatten(), torch.tensor(scalar))
        self.assertFalse(result.requires_grad)
        self.assertIsNone(model.slope.grad)
        # A nonconstant field must be re-evaluated along its own trajectory.
        self.assertNotAlmostEqual(float(result[0]), 0.6 * 2 + 0.2)

    def test_short_and_zero_intervals_keep_fp32_velocity(self):
        model = AffineVelocity(0, 0.0123)
        x = torch.full((2, 50, 32), 100, dtype=torch.bfloat16)
        start = torch.tensor([0.500001, 0.5])
        end = torch.tensor([0.5, 0.5])
        result = mfd.average_velocity(model, x, start, end, None, steps=1)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.full_like(result, 0.0123))

    def test_student_sign_mask_and_no_teacher_auxiliary_gradient(self):
        generated = torch.zeros(2, 50, 32, requires_grad=True)
        teacher, fake = AffineVelocity(0, 1), AffineVelocity(0, 0)
        start, end = torch.full((2,), 0.5), torch.full((2,), 0.2)
        loss, parts = mfd.student_loss(
            generated, teacher, fake, None, intervals=(start, end, 3)
        )
        loss.backward()
        # Positive teacher-fake velocity means a descent step lowers the action.
        torch.testing.assert_close(
            generated.grad[..., :14], torch.full((2, 50, 14), 0.5 / (2 * 50 * 14))
        )
        self.assertEqual(torch.count_nonzero(generated.grad[..., 14:]), 0)
        self.assertIsNone(teacher.slope.grad)
        self.assertIsNone(fake.slope.grad)
        self.assertEqual(float(parts["mfd_guidance_nfe"]), 6)

    def test_identical_models_zero_direction_and_single_step_limit(self):
        generated = torch.randn(3, 50, 32, requires_grad=True)
        model = AffineVelocity(0.4, 0.3)
        start, end = torch.full((3,), 0.7), torch.full((3,), 0.1)
        loss, parts = mfd.student_loss(
            generated, model, model, None, intervals=(start, end, 4)
        )
        self.assertEqual(float(parts["mfd_direction_rms"]), 0)
        loss.backward()
        self.assertEqual(torch.count_nonzero(generated.grad), 0)
        x = torch.randn(3, 50, 32)
        torch.testing.assert_close(
            mfd.average_velocity(model, x, start, end, None, steps=1),
            model(x, start, None),
        )

    def test_auxiliary_is_velocity_regression_and_detaches_student(self):
        generated = torch.zeros(2, 50, 32, requires_grad=True)
        fake = AffineVelocity(0.2)
        with (
            patch.object(torch, "rand", return_value=torch.full((2,), 0.5)),
            patch.object(torch, "randn_like", return_value=torch.ones_like(generated)),
        ):
            loss, parts = mfd.auxiliary_loss(fake, generated, None)
        # x=.5, predicted velocity=.1, target=1: MSE=.81, without (1-t)^2.
        self.assertAlmostEqual(float(loss), 0.81, places=5)
        torch.testing.assert_close(
            loss.detach(),
            (parts["fake_physical_mse"] * 14 + parts["fake_padding_mse"] * 18) / 32,
        )
        loss.backward()
        self.assertIsNone(generated.grad)
        self.assertIsNotNone(fake.slope.grad)

    def test_interval_rng_replay(self):
        state = torch.get_rng_state()
        first = mfd.sample_intervals(192, "cpu")
        torch.set_rng_state(state)
        second = mfd.sample_intervals(192, "cpu")
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_config_inherits_accepted_dmd_model_and_budget(self):
        base = {
            "method": "onedp",
            "batch_size": 384,
            "microbatch_per_rank": 192,
            "auxiliary_updates_per_student": 5,
            "onedp_loss_action_dim": 14,
            "lr": 1e-5,
            "aux_lr": 2e-4,
            "data_manifest": "accepted-manifest.json",
            "epochs": 20,
            "model": {"openpi": {"action_chunk": 50}},
        }
        derived = config_from_dmd(base)
        self.assertEqual(derived["model"], base["model"])
        self.assertEqual(derived["epochs"], 20)
        self.assertEqual(derived["method"], "mfd")
        self.assertEqual(base["method"], "onedp")
        self.assertEqual(derived["mfd_loss_action_dim"], 14)
        derived["model"]["openpi"]["action_chunk"] = 20
        self.assertEqual(base["model"]["openpi"]["action_chunk"], 50)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
