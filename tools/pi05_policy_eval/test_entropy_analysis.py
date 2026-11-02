import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entropy_analysis import comparison, entropy, executed_count


def test_saved_observation_omits_disabled_camera(tmp_path):
    from teacher_replay import load_observation

    p = tmp_path / "obs.npz"
    np.savez(
        p,
        main_images=np.zeros((2, 3, 3, 3), dtype=np.uint8),
        wrist_images=np.zeros((2, 2, 3, 3, 3), dtype=np.uint8),
        states=np.zeros((2, 14), dtype=np.float32),
        task_descriptions=np.array(["a", "b"]),
    )
    obs = load_observation(p)
    assert obs["extra_view_images"] is None
    assert obs["task_descriptions"] == ["a", "b"]


def test_one_prediction_cannot_measure_policy_entropy():
    with pytest.raises(ValueError):
        entropy(np.zeros((2, 1, 50, 14)))


def test_sample_axis_real_dimensions_and_scale():
    a = np.random.default_rng(12).normal(size=(2, 4, 50, 32))
    h = entropy(a)
    b = a.copy()
    b[..., 14:] += 1e8
    np.testing.assert_array_equal(h, entropy(b))
    np.testing.assert_allclose(entropy(a[:, ::-1]), h)
    assert np.all(entropy(a * 2) > h)


def test_fixed_bins_overlap_ties_and_early_success_mask():
    x = np.arange(100.0)
    same = comparison(x, x)
    assert same["overlap"] == 1 and all(b["delta"] == 0 for b in same["bins"])
    assert comparison(x, -x)["overlap"] == 0
    assert comparison(np.ones(100), np.ones(100))["overlap"] == 1
    assert executed_count({(25, 4): 15.2}, 25, 4, 80, 100, True) == 17
    assert executed_count({(25, 4): 100}, 25, 4, 80, 100, True) == 20
    assert executed_count({}, 25, 4, 80, 100, False) == 0
