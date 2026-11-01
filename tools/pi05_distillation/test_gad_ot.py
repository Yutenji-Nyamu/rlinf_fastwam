# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Assignment/gradient/RNG checks for conditional endpoint GAD-OT."""

import copy
import unittest

import torch

from rlinf.algorithms.pi05_distillation import gad, losses, ot
from rlinf.algorithms.pi05_distillation.resume_extension import validate_resume


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 32))

    def forward(self, x, t, context):
        return 0.2 * x + self.bias

    @torch.no_grad()
    def endpoint(self, z, context, steps=5):
        x = z.clone()
        for _ in range(steps):
            x = x - self(x, None, None) / steps
        return x


class Checks(unittest.TestCase):
    def test_permutations_stay_within_observation(self):
        teacher = torch.arange(8.0).reshape(2, 4, 1, 1).expand(2, 4, 3, 14)
        permutation = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
        student = teacher.gather(1, permutation[:, :, None, None].expand_as(teacher))
        actual, metrics = ot.match(student, teacher, torch.ones(2, 3))
        torch.testing.assert_close(actual, permutation)
        self.assertEqual(metrics["ot_matched_weighted_mse"].item(), 0)
        self.assertGreater(metrics["ot_cost_reduction"].item(), 0)

    def test_one_whole_chunk_not_independent_action_matching(self):
        teacher = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]])[..., None].expand(
            1, 2, 2, 14
        )
        student = torch.tensor([[[-1.0, 1.0], [1.0, -1.0]]])[..., None].expand_as(
            teacher
        )
        identity, metrics = ot.match(student, teacher, torch.ones(1, 2))
        torch.testing.assert_close(identity, torch.tensor([[0, 1]]))
        self.assertEqual(metrics["ot_matched_weighted_mse"].item(), 2)
        left, _ = ot.match(student, teacher, torch.tensor([[1.5, 0.5]]))
        right, _ = ot.match(student, teacher, torch.tensor([[0.5, 1.5]]))
        torch.testing.assert_close(left, torch.tensor([[0, 1]]))
        torch.testing.assert_close(right, torch.tensor([[1, 0]]))

    def test_importance_does_not_change_assignment(self):
        torch.manual_seed(3)
        s, t = torch.randn(3, 4, 5, 14), torch.randn(3, 4, 5, 14)
        w = torch.rand(3, 5) + 0.5
        a, _ = ot.match(s, t, w)
        b, _ = ot.match(s, t, w * torch.tensor([[0.3], [1.2], [1.9]]))
        torch.testing.assert_close(a, b)
        for row in b.tolist():
            self.assertEqual(sorted(row), list(range(4)))

    def test_retaining_candidates_preserves_original_signals_and_rng(self):
        s, t = Toy(), Toy()
        actions = torch.zeros(3, 5, 32)
        before = torch.get_rng_state()
        original = gad.observe(s, t, actions, None, teacher_steps=5, seed=77)
        retained = gad.observe(
            s, t, actions, None, teacher_steps=5, seed=77, retain_candidates=True
        )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for a, b in zip(original[:2], retained[:2]):
            torch.testing.assert_close(a, b)
        for name in original[2]:
            torch.testing.assert_close(original[2][name], retained[2][name])
        self.assertTrue(all(not x.requires_grad for x in retained[3].values()))
        n1, target1, _ = ot.select(retained[3], retained[0], seed=101)
        n2, target2, _ = ot.select(retained[3], retained[0], seed=101)
        torch.testing.assert_close(n1, n2)
        torch.testing.assert_close(target1, target2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_k1_matches_direct_gradient_and_padding_mask(self):
        s, t = Toy(), Toy()
        z = torch.randn(3, 5, 32)
        w = (torch.rand(3, 5) + 0.5).requires_grad_()
        with torch.no_grad():
            prediction = (z - s(z, None, None))[..., :14]
            target = t.endpoint(z, None, 5)[..., :14]
        target.requires_grad_()
        candidates = {
            "noise": z[:, None],
            "student": prediction[:, None],
            "teacher": target[:, None],
        }
        direct, _ = losses.endpoint_loss(s, t, z, None, teacher_steps=5, weights=w)
        direct.backward()
        reference = s.bias.grad.clone()
        s.zero_grad()
        loss, _ = ot.endpoint_loss(s, candidates, None, w, seed=102)
        loss.backward()
        torch.testing.assert_close(loss, direct)
        torch.testing.assert_close(s.bias.grad, reference)
        self.assertEqual(s.bias.grad[14:].abs().max().item(), 0)
        self.assertIsNone(w.grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(t.bias.grad)

    def test_row_mean_keeps_loss_scale(self):
        s, t = Toy(), Toy()
        w, _, _, candidates = gad.observe(
            s,
            t,
            torch.zeros(2, 3, 32),
            None,
            teacher_steps=5,
            seed=99,
            retain_candidates=True,
        )
        assignment, metrics = ot.match(candidates["student"], candidates["teacher"], w)
        batch = torch.arange(2)
        row_losses = []
        for row in range(4):
            z = candidates["noise"][:, row]
            pred = (z - s(z, None, None))[..., :14]
            target = candidates["teacher"][batch, assignment[:, row]]
            row_losses.append(gad.weighted_mean((pred - target).square(), w))
        torch.testing.assert_close(
            torch.stack(row_losses).mean(), metrics["ot_matched_weighted_mse"]
        )

    def test_explicit_contract_and_strict_resume(self):
        config = {
            "method": "endpoint",
            "gad": {"enabled": True},
            "gad_ot": copy.deepcopy(ot.CONTRACT),
            "loss_action_dim": 14,
            "teacher_steps": 5,
        }
        self.assertTrue(ot.validate_config(config))
        old = {k: v for k, v in config.items() if k != "gad_ot"}
        self.assertFalse(ot.validate_config(old))
        with self.assertRaises(ValueError):
            ot.validate_config(dict(config, method="cp_v2"))
        with self.assertRaises(ValueError):
            validate_resume(
                {"config": old, "windows": 7, "world_size": 2}, config, 7, 2
            )
        self.assertIsNone(
            validate_resume(
                {"config": config, "windows": 7, "world_size": 2}, config, 7, 2
            )
        )

    def test_invalid_cost_does_not_reach_training(self):
        a = torch.zeros(1, 4, 2, 14)
        with self.assertRaises(FloatingPointError):
            ot.match(a, a, torch.zeros(1, 2))
        a[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            ot.match(a, a, torch.ones(1, 2))


if __name__ == "__main__":
    unittest.main()
