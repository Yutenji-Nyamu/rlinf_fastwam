"""The full-noise boundary must not silently discard positions 1..49."""
import pytest
import torch

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import project_dsrl_trajectory
from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy


def test_replay_retains_every_full_noise_position():
    actions = torch.arange(3200).reshape(1, 2, 50, 32).float() / 1000
    done = torch.zeros(1, 2, 50, dtype=torch.bool)
    images = torch.zeros(1, 2, 64, 64, 3, dtype=torch.uint8)
    obs = {"main_images": images, "states": torch.zeros(1, 2, 14)}
    trajectory = Trajectory(actions=actions, rewards=torch.zeros(1, 2, 50),
        terminations=done, truncations=done, dones=done, curr_obs=obs, next_obs=obs)
    kwargs = dict(action_horizon=50, state_dim=14, num_action_chunks=50, gamma=.999)
    full = project_dsrl_trajectory(trajectory, latent_dim=1600, full_noise=True, **kwargs)
    assert torch.equal(full["actions"], actions.reshape(2, 1600).bfloat16())
    with pytest.raises((ValueError, AssertionError)):
        project_dsrl_trajectory(trajectory, latent_dim=32, **kwargs)


def test_full_gaussian_joint_density_and_gradient_are_finite():
    actor = GaussianPolicy(128, 1600, action_horizon=1,
        low=torch.full((1600,), -1.), high=torch.ones(1600)).bfloat16()
    actor.probability_float32 = True
    actions, logp = actor.sample(torch.randn(8, 128).bfloat16())
    assert actions.shape == (8, 1, 1600)
    assert logp.dtype == torch.float32 and torch.isfinite(logp).all()
    noise = actions.reshape(8, 50, 32)
    assert not torch.equal(noise[:, 0], noise[:, 1])
    (actions.square().mean() + .02 * logp.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
