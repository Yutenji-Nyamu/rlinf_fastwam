import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_shadow import compare, observe
from noise_samples import collect_noise_samples


class FakePolicy:
    def predict_action_batch(self, obs, **kw):
        z = kw["noise"]
        y = z[..., :14] + obs["states"][:, None, :14]
        return y, {"dvac_telemetry": {"final_model_action": y, "initial_noise_full": z}}


def test_same_noise_teacher_distance_and_padding_excluded():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(2, 4, 50, 32)).astype(np.float32)
    teacher = a[:, 0].copy()
    teacher[..., 14:] += 10000
    scores = compare(a, teacher)
    assert scores["teacher_student_rmse"].shape == (2, 50)
    assert not scores["teacher_student_rmse"].any()
    teacher[..., :14] += 2
    np.testing.assert_allclose(
        compare(a, teacher)["teacher_student_rmse"], 2, atol=1e-6
    )


def test_extra_batch_order_and_main_output_preserved():
    z = torch.zeros(4, 50, 32)
    obs = {
        "states": torch.arange(4.0)[:, None].repeat(1, 14),
        "task_descriptions": ["a"] * 4,
    }
    main = obs["states"][:, None].repeat(1, 50, 1)
    trace = {"initial_noise_full": z, "final_model_action": main}
    state = torch.get_rng_state()
    full = collect_noise_samples(FakePolicy(), obs, trace, main, 22)
    split = collect_noise_samples(FakePolicy(), obs, trace, main, 22, max_extra_batch=6)
    for key in full:
        np.testing.assert_array_equal(full[key], split[key])
    np.testing.assert_array_equal(full["noise4_env_actions"][:, 0], main.numpy())
    assert torch.equal(state, torch.get_rng_state())


def test_shadow_with_cpu_environment_actions_preserves_rng():
    z = torch.zeros(4, 50, 32)
    obs = {
        "states": torch.arange(4.0)[:, None].repeat(1, 14),
        "task_descriptions": ["a"] * 4,
    }
    main = obs["states"][:, None].repeat(1, 50, 1)
    trace = {"initial_noise_full": z, "final_model_action": main}
    state = torch.get_rng_state()
    result = observe(
        FakePolicy(), FakePolicy(), obs, trace, main, 22, max_extra_batch=6
    )
    assert not result["teacher_student_rmse"].any()
    assert torch.equal(state, torch.get_rng_state())
