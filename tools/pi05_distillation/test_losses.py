# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Targeted CPU checks for flow-coordinate signs and consistency gradients."""

import unittest

import torch

from rlinf.algorithms.pi05_distillation.losses import (
    cp_distance,
    cp_loss,
    flow_map,
    onedp_loss,
    fake_loss,
)
from rlinf.algorithms.pi05_distillation.model import load_expert


class ScalarVelocity(torch.nn.Module):
    def __init__(self, value, train=True):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)), requires_grad=train)

    def forward(self, x, t, context, stop=None):
        return self.value * x + 0.3


class DistillationTests(unittest.TestCase):
    def test_cp_distance_matches_reference_and_keeps_small_errors(self):
        pred = torch.randn(3, 50, 32)
        target = torch.randn_like(pred)
        delta = (50 * 32) ** 0.5 * 0.00054
        reference = ((pred - target).square().square() + delta**2).sqrt() - delta
        self.assertTrue(torch.allclose(cp_distance(pred, target), reference.mean()))
        small = torch.full((2, 50, 32), 1e-4, requires_grad=True)
        loss = cp_distance(small, torch.zeros_like(small))
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertTrue((small.grad > 0).all())

    def test_consistency_input_gradient_and_frozen_teacher(self):
        torch.manual_seed(17)
        student, teacher, ema = (
            ScalarVelocity(0.4),
            ScalarVelocity(0.8, False),
            ScalarVelocity(0.7, False),
        )
        loss, parts = cp_loss(student, teacher, ema, torch.randn(8, 50, 32), None)
        # Test consistency itself, not only the supervised anchor.
        consistency_grad = torch.autograd.grad(
            parts["consistency"], student.value, retain_graph=True
        )[0]
        self.assertTrue(torch.isfinite(consistency_grad))
        self.assertGreater(abs(float(consistency_grad)), 0)
        loss.backward()
        self.assertGreater(abs(float(student.value.grad)), 0)
        self.assertIsNone(teacher.value.grad)
        self.assertIsNone(ema.value.grad)

    def test_flow_boundary(self):
        x, t = torch.randn(4, 50, 32), torch.rand(4)
        self.assertTrue(torch.equal(x, flow_map(ScalarVelocity(0.4), x, t, t, None)))

    def test_onedp_score_sign_and_chain_factor(self):
        action = torch.zeros(2, 3, 4, requires_grad=True)

        def teacher(x, t, c):
            return torch.ones_like(x)

        def fake(x, t, c):
            return torch.zeros_like(x)

        loss, _ = onedp_loss(action, teacher, fake, None, time_min=0.5, time_max=0.5)
        loss.backward()
        # t*(1-t)^2 / number of scalars; positive gradient lowers the action.
        self.assertTrue(
            torch.allclose(action.grad, torch.full_like(action, 0.125 / action.numel()))
        )

    def test_overlay_rejects_missing_expert(self):
        model = torch.nn.Module()
        model.action_in_proj = torch.nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            load_expert(model, {})

    def test_onedp_physical_mask_gradient_and_full_auxiliary_fit(self):
        generated = torch.zeros(2, 50, 32, requires_grad=True)

        def teacher(x, t, context):
            return torch.ones_like(x)

        def fake(x, t, context):
            return torch.zeros_like(x)

        loss, _ = onedp_loss(
            generated, teacher, fake, None,
            time_min=0.5, time_max=0.5, action_dim=14,
        )
        loss.backward()
        torch.testing.assert_close(
            generated.grad[..., :14],
            torch.full_like(generated.grad[..., :14], 0.125 / (2 * 50 * 14)),
        )
        self.assertEqual(torch.count_nonzero(generated.grad[..., 14:]), 0)
        auxiliary = ScalarVelocity(0.3)
        lf, parts = fake_loss(auxiliary, generated, None, return_parts=True)
        torch.testing.assert_close(
            lf.detach(), (parts['fake_physical_mse'] * 14 + parts['fake_padding_mse'] * 18) / 32,
        )
        lf.backward()
        self.assertTrue(torch.isfinite(auxiliary.value.grad))


if __name__ == "__main__":
    unittest.main()
