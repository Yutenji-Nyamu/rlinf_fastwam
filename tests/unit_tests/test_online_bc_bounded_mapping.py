# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch

from rlinf.algorithms.online_bc_dvac import OnlineBCDvac, log_moments
from rlinf.data.online_bc import SuccessReplay, masked_fm_loss


def row(z):
    return {"dvac_v": torch.as_tensor(z, dtype=torch.float64).exp(),
            "action_valid_mask": torch.ones(len(z), 14, dtype=torch.bool)}


def mapper(lo=0, hi=5):
    d = OnlineBCDvac(mapping="bounded_linear", weight_min=lo, weight_max=hi)
    d.history.append(torch.tensor([2., 0., 2.], dtype=torch.float64))  # mu=0, std=1
    return d


def test_endpoint_mapping_no_chunk_centering_and_masked_fm():
    d = mapper()
    r = row([-100, -2, -1, 0, 1, 2, 100])
    d.annotate([[r]], torch.zeros(3))
    expected = torch.tensor([0., 0., .5, 1., 3., 5., 5.])
    torch.testing.assert_close(r["action_weights"], expected, atol=1e-6, rtol=1e-6)
    positive, negative = row([1.] * 50), row([-1.] * 50)
    positive["action_valid_mask"][:4] = False
    d.annotate([[positive, negative]], torch.zeros(3))
    torch.testing.assert_close(positive["action_weights"], torch.full((50,), 3.))
    torch.testing.assert_close(negative["action_weights"], torch.full((50,), .5))
    loss = torch.ones(1, 50, 14, requires_grad=True)
    result = masked_fm_loss(loss, positive["action_valid_mask"][None],
                            positive["action_weights"][None])
    result.backward()
    torch.testing.assert_close(result, torch.tensor(3.))
    assert loss.grad[:, :4].eq(0).all()
    assert not positive["action_weights"].requires_grad


def test_cold_start_freeze_and_exact_restore():
    d = OnlineBCDvac(mapping="bounded_linear", weight_min=0, weight_max=5)
    first = row([-1, 0, 1])
    moments = log_moments(first["dvac_v"], 1e-12)
    d.annotate([[first]], moments)
    assert first["action_weights"].eq(1).all()
    state = copy.deepcopy(d.state_dict())
    restored = OnlineBCDvac(mapping="bounded_linear", weight_min=0, weight_max=5)
    restored.load_state_dict(state)
    a, b = row([-3, 0, 3]), row([-3, 0, 3])
    assert d.annotate([[a]], moments) == restored.annotate([[b]], moments)
    assert torch.equal(a["action_weights"], b["action_weights"])
    assert first["action_weights"].eq(1).all()
    with pytest.raises(ValueError):
        OnlineBCDvac().load_state_dict(state)
    # Legacy centered checkpoint settings stay compatible and byte-equivalent.
    legacy = OnlineBCDvac().state_dict()
    assert "mapping" not in legacy["settings"]
    OnlineBCDvac().load_state_dict(legacy)


def test_limit_does_not_change_dvac_calibration_or_kept_weights(tmp_path):
    a, b = mapper(), mapper()
    episodes = [[row([1, 2])] * 3, [row([-1, -2])] * 4]
    copied = copy.deepcopy(episodes)
    moments = sum((log_moments(r["dvac_v"], 1e-12) for ep in episodes for r in ep),
                  torch.zeros(3, dtype=torch.float64))
    a.annotate(episodes, moments)
    b.annotate(copied, moments)
    pool = SuccessReplay(42, str(tmp_path / "pool"), max_success_chunks=3)
    pool.add_episodes(episodes)
    assert pool.episodes == 1 and len(pool) == 3
    assert torch.equal(a.history[-1], b.history[-1])
    assert torch.equal(pool.records[0]["action_weights"], copied[0][0]["action_weights"])


def test_unit_endpoints_recover_plain_bc():
    d = mapper(1, 1)
    r = row([-100, 0, 100])
    d.annotate([[r]], torch.zeros(3))
    assert r["action_weights"].eq(1).all()


@pytest.mark.parametrize("lo,hi", [(None, 5), (0, None), (-1, 5), (2, 5), (0, .5), (0, float("nan"))])
def test_invalid_endpoints(lo, hi):
    with pytest.raises(ValueError):
        OnlineBCDvac(mapping="bounded_linear", weight_min=lo, weight_max=hi)
