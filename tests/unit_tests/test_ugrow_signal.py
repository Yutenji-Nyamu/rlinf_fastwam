"""Numerical checks against the independently specified two-budget formula."""

import math

import pytest
import torch

from rlinf.algorithms.ugrow_signal import compute_ugrow_signal


def test_ugrow_matches_scalar_reference_and_ignores_padding():
    a = torch.linspace(-3.0, 4.0, 2 * 7 * 32).reshape(2, 7, 32)
    b = a.sin() * 2
    actual = compute_ugrow_signal(a.requires_grad_(), b.requires_grad_())
    reference = torch.tensor([
        [sum(
            abs(float(a[i, h, d]) / 2 - float(b[i, h, d]) / 2)
            / (math.hypot(float(a[i, h, d]) / math.sqrt(2),
                          float(b[i, h, d]) / math.sqrt(2)) + 1e-8)
            for d in range(14)
        ) / 14 for h in range(7)]
        for i in range(2)
    ])
    torch.testing.assert_close(actual, reference)
    assert actual.dtype == torch.float32 and not actual.requires_grad
    changed = b.detach().clone()
    changed[..., 14:] = float("nan")
    torch.testing.assert_close(compute_ugrow_signal(a, changed), reference)


def test_zero_equal_opposite_and_extreme_finite_actions():
    a = torch.zeros(1, 4, 14, dtype=torch.float64)
    a[:, 1] = 2
    a[:, 2] = 1e300
    a[:, 3] = 1e-300
    b = a.clone()
    b[:, 1:3] = -b[:, 1:3]
    u = compute_ugrow_signal(a, b)
    torch.testing.assert_close(u, torch.tensor([[0., 1., 1., 0.]]))


@pytest.mark.parametrize("kind", ["nan", "negative_eps", "wrong_shape", "int"])
def test_bad_signal_inputs_fail_closed(kind):
    a = torch.zeros(1, 2, 32)
    b = a.clone()
    kwargs = {}
    if kind == "nan":
        b[0, 0, 0] = float("nan")
    elif kind == "negative_eps":
        kwargs["eps"] = -1.0
    elif kind == "wrong_shape":
        b = b[:, :1]
    else:
        b = b.long()
    with pytest.raises(ValueError):
        compute_ugrow_signal(a, b, **kwargs)
