import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observe import derive, observe


class FakePolicy:
    def _get_timesteps(self, n, device):
        return torch.linspace(1, 0, n + 1, device=device)

    def predict_action_batch(self, obs, **kw):
        n, z = kw["num_steps"], kw["noise"]
        torch.rand(3)  # The observer must undo even sampler-internal RNG draws.
        y = z[..., :14] / n + obs["states"][:, None, :14]
        trace = dict(
            initial_noise_full=z, final_model_action=y,
            x_chain=torch.stack([z[..., :14] + i for i in range(n + 1)], 1),
            z_endpoint=torch.stack([y + i / n for i in range(n)], 1),
            timesteps=self._get_timesteps(n, z.device)[:-1],
        )
        return y.cpu(), {"dvac_telemetry": trace}


def sample():
    z = torch.arange(2 * 50 * 32, dtype=torch.float32).reshape(2, 50, 32) / 1000
    obs = dict(states=torch.arange(2.0)[:, None].repeat(1, 14), task_descriptions=["a", "b"])
    model = FakePolicy()
    a, result = model.predict_action_batch(obs, num_steps=10, noise=z)
    return model, obs, a, result["dvac_telemetry"]


def test_shared_noise_full_solvers_batch_order_and_rng():
    model, obs, action, trace = sample()
    before = torch.get_rng_state()
    full = observe(model, obs, trace, action, 123)
    split = observe(model, obs, trace, action, 123, max_extra_batch=3)
    assert torch.equal(before, torch.get_rng_state())
    for key in full:
        np.testing.assert_array_equal(full[key], split[key])
    np.testing.assert_array_equal(full["actions_env_n10"][:, 0], action.numpy())
    for n in (1, 5, 10):
        expected = full["noise4_initial_noise"][..., :14] / n + obs["states"].numpy()[:, None, None]
        np.testing.assert_array_equal(full[f"actions_model_n{n}"], expected)
        assert full[f"x_chain_n{n}"].shape == (2, n + 1, 50, 14)
        assert full[f"z_endpoint_n{n}"].shape == (2, n, 50, 14)
        assert full[f"final_time_n{n}"] == 0
    assert np.isnan(full["dv_n1"]).all() and not full["dv_valid_n1"].any()
    assert full["dv_valid_n5"].all() and full["dv_valid_n10"].all()


def test_raw_scores_reproduce_and_padding_cannot_change_dispersion():
    model, obs, action, trace = sample()
    data = observe(model, obs, trace, action, 123)
    scores = derive(data)
    for n in (1, 5, 10):
        raw = data[f"actions_model_n{n}"]
        data[f"actions_model_n{n}"] = np.pad(raw, [(0, 0)] * 3 + [(0, 18)], constant_values=1e9)
    again = derive(data)
    for key in scores:
        np.testing.assert_array_equal(scores[key], again[key])
    assert (scores["rmse_1_10"] > 0).all()
