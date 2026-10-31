# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Small CPU checks for GAD's math, baseline equivalence and resumable sampling."""

import copy
import unittest

import numpy as np
import torch

from rlinf.algorithms.pi05_distillation import gad, losses, mfd


class Toy(torch.nn.Module):
    def __init__(self, value=0.2):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(value))

    def forward(self, x, t, context, stop=None):
        return self.w * x + 0.1

    @torch.no_grad()
    def endpoint(self, noise, context, steps=5):
        x = noise.clone()
        for i in range(steps):
            x -= self(x, None, None) / steps
        return x


class Data:
    def __len__(self):
        return 32

    def batch(self, ids, executor=None):
        return np.asarray(ids)


class Checks(unittest.TestCase):
    def test_rank_ties_and_range(self):
        x = torch.tensor([3.0, 1.0, 1.0, 4.0])
        torch.testing.assert_close(
            gad.midranks(x), torch.tensor([0.625, 0.25, 0.25, 0.875])
        )
        s = torch.randn(12, 4, 50, 14, requires_grad=True)
        w, _, p = gad.action_weights(s, s * 0.5 + 0.4)
        self.assertFalse(w.requires_grad)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=6)
        self.assertTrue(w.min() >= 0.5 and w.max() <= 1.5)
        self.assertGreater(p["gad_weight_std"].item(), 0.2)
        zero, _, _ = gad.action_weights(torch.zeros_like(s), torch.zeros_like(s))
        torch.testing.assert_close(zero, torch.ones_like(zero))

    def test_pairing_and_rng(self):
        t = Toy()
        a = torch.randn(4, 50, 32)
        before = torch.get_rng_state()
        w1, p1, _ = gad.observe(t, t, a, None, teacher_steps=5, seed=19)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        w2, p2, _ = gad.observe(t, t, a, None, teacher_steps=5, seed=19)
        torch.testing.assert_close(w1, w2)
        torch.testing.assert_close(p1, p2)

    def test_all_losses_unit_weight_equivalence(self):
        actions = torch.randn(4, 6, 32)
        weights = torch.ones(4, 6)
        teacher, ema = Toy(0.3), Toy(0.25)
        for fn in ("endpoint", "cp", "fake", "onedp", "mfd_fake", "mfd"):
            grads, values = [], []
            for use in (None, weights):
                torch.manual_seed(99)
                student = Toy()
                if fn == "endpoint":
                    loss = losses.endpoint_loss(
                        student, teacher, actions, None, teacher_steps=5, weights=use
                    )[0]
                elif fn == "cp":
                    loss = losses.cp_loss(
                        student, teacher, ema, actions, None, action_dim=14, weights=use
                    )[0]
                elif fn == "fake":
                    loss = losses.fake_loss(student, actions, None, weights=use)
                elif fn == "mfd_fake":
                    loss = mfd.auxiliary_loss(student, actions, None, weights=use)[0]
                else:
                    generated = student(actions, None, None)
                    f = losses.onedp_loss if fn == "onedp" else mfd.student_loss
                    loss = f(generated, teacher, ema, None, action_dim=14, weights=use)[
                        0
                    ]
                loss.backward()
                grads.append(student.w.grad.clone())
                values.append(loss.detach())
            torch.testing.assert_close(values[0], values[1], rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(grads[0], grads[1], rtol=1e-5, atol=1e-7)

    def test_signed_gradient_and_detachment(self):
        value = torch.ones(1, 2, 14, requires_grad=True)
        w = torch.tensor([[0.5, 1.5]], requires_grad=True)
        loss = gad.weighted_mean(-value, w)
        loss.backward()
        self.assertIsNone(w.grad)
        torch.testing.assert_close(value.grad[0, 1], value.grad[0, 0] * 3)
        self.assertTrue((value.grad < 0).all())

    def test_sampling_resume_and_correction(self):
        a = gad.PriorityBatches(Data(), rank=0, world=1, batch=32, seed=7, start=0)
        try:
            _, draw = a.next(0)
            a.update(draw["ids"], torch.linspace(0, 1, 32))
            saved = copy.deepcopy(a.state_dict())
            b = gad.PriorityBatches(
                Data(), rank=0, world=1, batch=32, seed=7, start=1, state=saved
            )
            try:
                for step in (1, 2):
                    x, p = a.next(step)
                    y, q = b.next(step)
                    np.testing.assert_array_equal(x, y)
                    self.assertEqual(p, q)
                    np.testing.assert_allclose(
                        np.asarray(p["probability"]) * p["importance"], 1 / 32
                    )
                    self.assertLessEqual(max(p["importance"]), 2)
                    a.update(p["ids"], torch.ones(32) * 0.7)
                    b.update(q["ids"], torch.ones(32) * 0.7)
            finally:
                b.close()
        finally:
            a.close()


if __name__ == "__main__":
    unittest.main()
