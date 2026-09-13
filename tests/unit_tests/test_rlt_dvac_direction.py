"""Course boundary and scoped transition, independent of worker/device setup."""

import pytest

from rlinf.algorithms.rlt.dvac_direction import (
    actor_weight_progress,
    direction_anchor,
    direction_factors,
)


COURSE = {"enable": True, "warmup_updates": 20000, "ramp_updates": 50000}


@pytest.mark.parametrize(
    ("step", "expected"), [(0, 1), (20000, 1), (69998, 1), (69999, -1), (70000, -1)]
)
def test_hard_direction_and_original_progress_share_boundary(step, expected):
    local, chunk, metrics = direction_factors(step, {"enable": True}, COURSE)
    assert local == chunk == expected
    _, progress = actor_weight_progress(step, 20000, 50000)
    assert metrics["rlt_dvac_new/direction_anchor_reached"] == float(progress == 1)


@pytest.mark.parametrize("scope", ["both", "local", "chunk"])
@pytest.mark.parametrize(("step", "expected"), [(69999, 1), (70049, 0), (70099, -1)])
def test_soft_transition_preserves_unselected_layer(scope, step, expected):
    cfg = {"enable": True, "scope": scope, "transition_updates": 100}
    local, chunk, _ = direction_factors(step, cfg, COURSE)
    assert local == (expected if scope != "chunk" else 1)
    assert chunk == (expected if scope != "local" else 1)


def test_no_default_mutation_and_scaled_course_anchor():
    empty = {}
    assert direction_factors(100000, empty, empty) == (1.0, 1.0, {})
    assert empty == {}
    course = {"enable": True, "warmup_updates": 10000, "ramp_updates": 25000}
    assert direction_anchor({"enable": True}, course)["end_update_step"] == 34999
    assert direction_factors(35000, {"enable": True}, course)[:2] == (-1, -1)


@pytest.mark.parametrize("override", [{"scope": "none"}, {"anchor": "round"}, {"transition_updates": -1}, {"transition_updates": 0.5}])
def test_invalid_enabled_schedule(override):
    with pytest.raises(ValueError):
        direction_factors(70000, {"enable": True, **override}, COURSE)
