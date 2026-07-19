from __future__ import annotations

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.fastwam import fastwam_rl as rl


class _OfficialSchedule:
    def __init__(self, shift: float = 5.0, num_train_timesteps: int = 1000):
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def build_inference_schedule(
        self, num_inference_steps, device, dtype, shift_override=None
    ):
        shift = self.shift if shift_override is None else float(shift_override)
        u = torch.linspace(
            1.0,
            0.0,
            num_inference_steps + 1,
            device=device,
            dtype=torch.float32,
        )
        sigma = self._phi(u, shift)
        timesteps = sigma[:-1] * float(self.num_train_timesteps)
        deltas = sigma[1:] - sigma[:-1]
        return timesteps.to(dtype), deltas.to(dtype)

    @staticmethod
    def step(model_output, delta, sample):
        return sample + model_output * delta.to(sample)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.infer_action_scheduler = _OfficialSchedule()


def _velocity(model, *, x, raw_timestep, conditioning):
    del raw_timestep, conditioning
    return torch.ones_like(x) * model.scale


def test_official_shifted_grid_has_expected_flow_sde_stds():
    model = _TinyModel()
    schedule = rl.resolve_action_schedule(model, 10, sigma_shift=None)
    assert schedule.effective_shift == 5.0
    assert torch.all(schedule.deltas < 0)
    torch.testing.assert_close(
        schedule.next_normalized_timesteps,
        schedule.normalized_timesteps + schedule.deltas.float(),
    )

    stds = []
    x = torch.zeros(1, 1, 1)
    for index in range(10):
        _, std, _ = rl.flow_step_mean_std(
            x=x,
            velocity=x,
            normalized_timestep=schedule.normalized_timesteps[index],
            next_normalized_timestep=schedule.next_normalized_timesteps[index],
            signed_delta=schedule.deltas[index],
            noise_level=0.1,
        )
        stds.append(std.item())
    expected = torch.tensor(
        [
            0.100000,
            0.107916,
            0.079156,
            0.067193,
            0.060634,
            0.056614,
            0.053979,
            0.052016,
            0.049801,
            0.044544,
        ]
    )
    torch.testing.assert_close(torch.tensor(stds), expected, atol=5e-6, rtol=5e-5)


def test_eval_singleton_broadcast_and_training_randomness_are_separate():
    evaluation = rl.prepare_initial_action_latents(
        batch_size=4,
        action_horizon=3,
        action_dim=2,
        device="cpu",
        dtype=torch.float32,
        rand_device="cpu",
        seed=0,
        broadcast_singleton=True,
    )
    for index in range(1, 4):
        torch.testing.assert_close(evaluation[0], evaluation[index])

    generator = torch.Generator(device="cpu").manual_seed(7)
    training = rl.prepare_rollout_randomness(
        batch_size=4,
        action_horizon=3,
        action_dim=2,
        num_inference_steps=10,
        device="cpu",
        rand_device="cpu",
        generator=generator,
    )
    assert torch.unique(training.denoise_inds).numel() == 1
    assert not torch.equal(training.initial_latents[0], training.initial_latents[1])
    assert not torch.equal(training.sde_epsilon[0], training.sde_epsilon[1])


def test_observation_value_feature_uses_final_video_cache_value_mean():
    first_values = torch.full((2, 3, 4), -100.0)
    final_values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    features = rl._pool_observation_value_features(
        [{"v": first_values}, {"v": final_values}],
        batch_size=2,
        video_seq_len=3,
        expected_dim=4,
    )
    torch.testing.assert_close(features, final_values.mean(dim=1))

    with pytest.raises(ValueError, match="feature mismatch"):
        rl._pool_observation_value_features(
            [{"v": final_values}],
            batch_size=2,
            video_seq_len=3,
            expected_dim=5,
        )
    invalid_values = final_values.clone()
    invalid_values[0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="floating-point and finite"):
        rl._pool_observation_value_features(
            [{"v": invalid_values}],
            batch_size=2,
            video_seq_len=3,
            expected_dim=4,
        )


def test_recompute_logprob_optionally_returns_observation_value_feature(monkeypatch):
    features = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    conditioning = rl.ActionConditioning(
        context=torch.empty(2, 0, 0),
        context_mask=torch.empty(2, 0, dtype=torch.bool),
        video_kv_cache=[],
        attention_mask=torch.empty(0, 0),
        video_seq_len=0,
        observation_value_features=features,
    )
    observed_kwargs = []

    def _conditioning(*args, **kwargs):
        del args
        observed_kwargs.append(kwargs)
        return conditioning

    monkeypatch.setattr(rl, "build_action_conditioning", _conditioning)
    monkeypatch.setattr(rl, "predict_action_velocity", _velocity)
    model = _TinyModel()
    chains = torch.zeros(2, 11, 3, 2)
    denoise_inds = torch.tensor([0, 0])
    common = {
        "input_image": torch.zeros(2, 3, 16, 16),
        "text_context": torch.zeros(2, 1, 1),
        "text_context_mask": torch.ones(2, 1, dtype=torch.bool),
        "proprio": torch.zeros(2, 2),
        "chains": chains,
        "denoise_inds": denoise_inds,
        "action_horizon": 3,
        "num_inference_steps": 10,
        "sigma_shift": None,
        "noise_level": 0.1,
    }

    default_result = rl.recompute_logprob(model, **common)
    assert len(default_result) == 2
    assert observed_kwargs[-1]["include_observation_value_features"] is False

    logprob, entropy, actual_features = rl.recompute_logprob(
        model,
        **common,
        return_observation_value_features=True,
        expected_value_feature_dim=4,
    )
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(actual_features, features)
    assert observed_kwargs[-1]["include_observation_value_features"] is True
    assert observed_kwargs[-1]["expected_value_feature_dim"] == 4


@pytest.mark.parametrize("selected_k", range(10))
def test_rollout_and_replay_use_the_real_selected_transition(monkeypatch, selected_k):
    monkeypatch.setattr(rl, "predict_action_velocity", _velocity)
    monkeypatch.setattr(
        rl, "build_action_conditioning", lambda *args, **kwargs: object()
    )
    model = _TinyModel()
    initial = torch.zeros(2, 3, 2)
    denoise_inds = torch.tensor([selected_k, selected_k], dtype=torch.long)
    epsilon = torch.tensor(
        [
            [[0.5, -0.2], [0.1, 0.3], [-0.4, 0.8]],
            [[-0.7, 0.4], [0.9, -0.1], [0.2, -0.6]],
        ]
    )
    with torch.no_grad():
        rollout = rl.flow_sde_rollout(
            model,
            conditioning=object(),
            initial_latents=initial,
            num_inference_steps=10,
            sigma_shift=None,
            noise_level=0.1,
            deterministic=False,
            denoise_inds=denoise_inds,
            sde_epsilon=epsilon,
        )
    assert rollout.chains is not None
    assert rollout.prev_logprobs is not None
    torch.testing.assert_close(
        rollout.chains[:, : selected_k + 1],
        torch.zeros(2, selected_k + 1, 3, 2),
    )
    for index in range(selected_k + 2, 11):
        torch.testing.assert_close(
            rollout.chains[:, selected_k + 1], rollout.chains[:, index]
        )

    replay_logprob, entropy = rl.recompute_logprob(
        model,
        input_image=torch.zeros(2, 3, 16, 16),
        text_context=torch.zeros(2, 1, 1),
        text_context_mask=torch.ones(2, 1, dtype=torch.bool),
        proprio=torch.zeros(2, 2),
        chains=rollout.chains,
        denoise_inds=denoise_inds,
        action_horizon=3,
        num_inference_steps=10,
        sigma_shift=None,
        noise_level=0.1,
    )
    torch.testing.assert_close(replay_logprob, rollout.prev_logprobs)
    assert torch.isfinite(entropy).all()

    replay_logprob.sum().backward()
    assert model.scale.grad is not None
    assert model.scale.grad.abs().item() > 0
    with torch.no_grad():
        model.scale.add_(1e-3)
    changed_logprob, _ = rl.recompute_logprob(
        model,
        input_image=torch.zeros(2, 3, 16, 16),
        text_context=torch.zeros(2, 1, 1),
        text_context_mask=torch.ones(2, 1, dtype=torch.bool),
        proprio=torch.zeros(2, 2),
        chains=rollout.chains,
        denoise_inds=denoise_inds,
        action_horizon=3,
        num_inference_steps=10,
        sigma_shift=None,
        noise_level=0.1,
    )
    assert not torch.equal(changed_logprob, rollout.prev_logprobs)


def test_deterministic_path_is_the_official_scheduler_step(monkeypatch):
    monkeypatch.setattr(rl, "predict_action_velocity", _velocity)
    model = _TinyModel()
    with torch.no_grad():
        model.scale.fill_(0.25)
    initial = torch.randn(2, 3, 2)
    schedule = rl.resolve_action_schedule(model, 10, sigma_shift=3.0)
    expected = initial.clone()
    with torch.no_grad():
        for delta in schedule.deltas:
            expected = model.infer_action_scheduler.step(
                torch.full_like(expected, 0.25), delta, expected
            )
        actual = rl.flow_sde_rollout(
            model,
            conditioning=object(),
            initial_latents=initial,
            num_inference_steps=10,
            sigma_shift=3.0,
            noise_level=0.1,
            deterministic=True,
        )
    assert schedule.configured_shift == 3.0
    assert schedule.effective_shift == 3.0
    assert actual.chains is None
    assert actual.prev_logprobs is None
    torch.testing.assert_close(actual.actions, expected)


def test_resource_chunking_does_not_change_precomputed_randomness(monkeypatch):
    monkeypatch.setattr(rl, "predict_action_velocity", _velocity)
    model = _TinyModel()
    randomness = rl.prepare_rollout_randomness(
        batch_size=4,
        action_horizon=3,
        action_dim=2,
        num_inference_steps=10,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(23),
    )
    with torch.no_grad():
        full = rl.flow_sde_rollout(
            model,
            conditioning=object(),
            initial_latents=randomness.initial_latents,
            num_inference_steps=10,
            sigma_shift=None,
            noise_level=0.1,
            deterministic=False,
            denoise_inds=randomness.denoise_inds,
            sde_epsilon=randomness.sde_epsilon,
        )
        pieces = []
        for start in (0, 2):
            item = slice(start, start + 2)
            pieces.append(
                rl.flow_sde_rollout(
                    model,
                    conditioning=object(),
                    initial_latents=randomness.initial_latents[item],
                    num_inference_steps=10,
                    sigma_shift=None,
                    noise_level=0.1,
                    deterministic=False,
                    denoise_inds=randomness.denoise_inds[item],
                    sde_epsilon=randomness.sde_epsilon[item],
                )
            )
    torch.testing.assert_close(
        torch.cat([piece.actions for piece in pieces]), full.actions
    )
    torch.testing.assert_close(
        torch.cat([piece.chains for piece in pieces]), full.chains
    )
    torch.testing.assert_close(
        torch.cat([piece.prev_logprobs for piece in pieces]), full.prev_logprobs
    )
