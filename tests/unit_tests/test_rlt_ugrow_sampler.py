"""Execute the production sampler with a tiny deterministic velocity oracle."""

import ast
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.algorithms.ugrow_signal import compute_ugrow_signal


def _production_sampler():
    path = Path(__file__).resolve().parents[2] / (
        "rlinf/models/embodiment/openpi/openpi_action_model.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef)
               and node.name == "OpenPi0ForRLActionPrediction")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_sample_actions_with_prefix_cache")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = dict(torch=torch, random=random, np=np,
                     compute_ugrow_signal=compute_ugrow_signal)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method.name]


class _TinySampler:
    _sample_actions_with_prefix_cache = _production_sampler()

    def __init__(self, *, fail_side=False):
        self.config = SimpleNamespace(
            num_steps=10, action_horizon=50, action_dim=32, action_env_dim=14,
            action_chunk=10, joint_logprob=False, is_nft=False, ignore_last=False,
        )
        # Generated noise is fp32, explicitly supplied noise is converted by
        # the legacy path. Side reuse must preserve either actual main dtype.
        self.action_in_proj = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
        self.use_vlm_value = False
        self.starts, self.calls = {}, []
        self.fail_side = fail_side

    @staticmethod
    def sample_noise(shape, device):
        return torch.randn(shape, device=device)

    @staticmethod
    def _init_nft_state(*_args):
        return None

    @staticmethod
    def _update_nft_state(*_args):
        pass

    @staticmethod
    def get_logprob_norm(value, *_args):
        return torch.zeros_like(value)

    def sample_mean_var_val(self, x, idx, state, masks, cache, method, steps, values):
        assert method == "flow_ode"
        self.calls.append((steps, idx, id(cache)))
        if idx == 0:
            self.starts[steps] = x.clone()
        # Exercise preservation of all random streams even for exceptional
        # side solves; these values intentionally do not influence velocity.
        random.random()
        np.random.random()
        if self.fail_side and steps == 5 and idx == 2:
            raise RuntimeError("side oracle failure")
        velocity = (x + state[:, :1, None]).tanh() + (1 - idx / steps) * 0.1
        return x - velocity / steps, torch.zeros_like(x), x.new_zeros(x.shape[0], 1), velocity


def _run(enabled, *, supplied=False, fail_side=False):
    torch.manual_seed(712)
    random.seed(731)
    np.random.seed(432)
    sampler = _TinySampler(fail_side=fail_side)
    state = torch.full((2, 14), 0.4)
    prefix = torch.zeros(2, 3, 8)
    noise = torch.randn(2, 50, 32) if supplied else None
    outputs = sampler._sample_actions_with_prefix_cache(
        state, prefix, torch.ones(2, 3), object(), noise=noise,
        mode="eval", compute_values=False, collect_ugrow=enabled,
    )
    return outputs, sampler, (torch.get_rng_state(), random.getstate(), np.random.get_state())


@pytest.mark.parametrize("supplied", [False, True])
def test_main_action_rng_and_complete_initial_noise_preserved(supplied):
    plain, p, prng = _run(False, supplied=supplied)
    enabled, e, erng = _run(True, supplied=supplied)
    for key, value in plain.items():
        assert torch.equal(value, enabled[key]), key
    assert torch.equal(prng[0], erng[0]) and prng[1] == erng[1]
    assert prng[2][0] == erng[2][0] and prng[2][2:] == erng[2][2:]
    np.testing.assert_array_equal(prng[2][1], erng[2][1])
    assert len(p.calls) == 10 and len(e.calls) == 15
    assert [s for s, _, _ in e.calls] == [10] * 10 + [5] * 5
    assert len({cache for _, _, cache in e.calls}) == 1
    assert torch.equal(e.starts[10], e.starts[5])
    assert e.starts[10].shape == (2, 50, 32)
    assert e.config.num_steps == 10
    u = enabled["teacher_ugrow_u"]
    assert u.shape == (2, 50) and torch.isfinite(u).all() and (u > 0).any()


def test_exception_restores_side_rng():
    _, _, expected = _run(False)
    with pytest.raises(RuntimeError, match="side oracle failure"):
        _run(True, fail_side=True)
    assert torch.equal(torch.get_rng_state(), expected[0])
    assert random.getstate() == expected[1]
    np.testing.assert_array_equal(np.random.get_state()[1], expected[2][1])
