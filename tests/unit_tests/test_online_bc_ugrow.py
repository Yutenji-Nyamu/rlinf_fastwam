"""U producer, successful replay, frozen calibration and FM contract checks."""

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.algorithms.online_bc_dvac import OnlineBCDvac, log_moments
from rlinf.algorithms.ugrow_signal import UGROW_SIGNAL_SPEC, compute_ugrow_signal
from rlinf.data.online_bc import SuccessEpisodeCollector, SuccessReplay, masked_fm_loss


def mapper():
    return OnlineBCDvac(
        signal_kind="ugrow_10_5", mapping="bounded_linear", weight_min=0, weight_max=5,
    )


def collect(u):
    c = SuccessEpisodeCollector(2, dvac_log_eps=1e-12, signal_kind="ugrow_10_5")
    u = torch.as_tensor(u, dtype=torch.float32).reshape(2, -1)
    obs = {"observation/state": torch.zeros(2, 14), "ugrow_u": u}
    commands = torch.full((2, u.shape[1], 14), 0.2)
    c.append(obs, commands, [True, False], torch.ones(2, 1), torch.tensor([3, 3]))
    # A post-terminal query must not add either moments or training records.
    c.append(obs, commands, [True, True], torch.ones(2, 1))
    return c.drain(), c.drain_dvac_moments()


def test_ugrow_matches_offline_numpy_formula_and_ignores_padding():
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=(2, 50, 32)), rng.normal(size=(2, 50, 32))
    expected = (np.abs(a[..., :14] / 2 - b[..., :14] / 2) / (
        np.hypot(a[..., :14] / np.sqrt(2), b[..., :14] / np.sqrt(2)) + 1e-8
    )).mean(-1)
    a[..., 14:] = np.nan
    u = compute_ugrow_signal(torch.tensor(a, requires_grad=True), torch.tensor(b))
    np.testing.assert_allclose(u.numpy(), expected, rtol=1e-6, atol=1e-7)
    assert u.dtype == torch.float32 and not u.requires_grad
    zeros = torch.zeros(1, 2, 14)
    assert compute_ugrow_signal(zeros, zeros).eq(0).all()
    huge = torch.full((1, 2, 14), 1e308, dtype=torch.float64)
    assert compute_ugrow_signal(huge, -huge).eq(1).all()
    with pytest.raises(ValueError):
        compute_ugrow_signal(zeros, torch.full_like(zeros, float("nan")))
    with pytest.raises(ValueError):
        compute_ugrow_signal(zeros, zeros, eps=0)


def test_native_sampler_same_main_action_noise_cache_rng_and_10_plus_5_calls():
    from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0ForRLActionPrediction

    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.action_in_proj = torch.nn.Linear(32, 2).to(dtype=torch.bfloat16)
    model.config = SimpleNamespace(
        num_steps=10, action_horizon=50, action_dim=32, action_chunk=50,
        action_env_dim=14, joint_logprob=False, is_nft=False,
    )
    model.use_vlm_value = False
    model._preprocess_observation = lambda obs, train: (None, None, None, None, obs.state)
    prefix_calls, calls, initial = [], [], []
    cache = {"prefix": torch.ones(2, 3)}

    def prefix(*args):
        prefix_calls.append(1)
        return None, None, cache

    def noise(shape, device):
        random.random()
        return torch.randn(shape, device=device)

    def velocity(x, idx, state, masks, seen_cache, method, steps, compute_values):
        assert seen_cache is cache and method == "flow_ode"
        calls.append(steps)
        if idx == 0:
            initial.append(x.clone())
        v = x * 0.2 + idx * 0.1
        return x - v / steps, torch.zeros_like(x), torch.zeros(x.shape[0]), v

    model._build_prefix_cache = prefix
    model.sample_noise = noise
    model.get_logprob_norm = lambda x, mean, std: torch.zeros_like(x)
    model._init_nft_state = lambda *args: {}
    model._update_nft_state = lambda *args: None
    model.sample_mean_var_val = velocity
    observation = SimpleNamespace(state=torch.zeros(2, 32))
    outputs, states = [], []
    for enabled in (False, True):
        calls.clear()
        initial.clear()
        prefix_calls.clear()
        torch.manual_seed(42)
        random.seed(42)
        outputs.append(model.sample_actions(observation, mode="eval", ugrow_enabled=enabled))
        states.append((torch.get_rng_state().clone(), random.getstate()))
        assert calls == [10] * 10 + ([5] * 5 if enabled else [])
        assert len(prefix_calls) == 1
        assert initial[0].dtype == torch.bfloat16
        if enabled:
            assert torch.equal(initial[0], initial[1])
    for key, value in outputs[0].items():
        assert torch.equal(value, outputs[1][key]), key
    assert torch.equal(states[0][0], states[1][0]) and states[0][1] == states[1][1]
    assert outputs[1]["ugrow_u"].shape == (2, 50)
    assert outputs[1]["ugrow_u"].max() > 0
    assert model.config.num_steps == 10 and cache["prefix"].eq(1).all()


def test_success_only_u_moments_frozen_weights_and_real_fm_gradient(tmp_path):
    first, moments = collect([[0.01, 0.1, 0.5], [0.02, 0.2, 0.8]])
    assert len(first) == 1 and moments[0] == 6
    torch.testing.assert_close(moments, log_moments(torch.tensor([0.01, 0.1, 0.5, 0.02, 0.2, 0.8]), 1e-12))
    m = mapper()
    assert m.annotate(first, moments)["ugrow/calibrated"] == 0
    assert first[0][0]["action_weights"].eq(1).all()
    second, second_moments = collect([[0, 0.1, 1], [0.1, 0.2, 0.3]])
    metrics = m.annotate(second, second_moments)
    record = second[0][0]
    mean, std = moments[1] / 6, (moments[2] / 6 - (moments[1] / 6).square()).sqrt()
    z = ((record["ugrow_u"].double().add(1e-12).log() - mean) / std).clamp(-2, 2)
    expected = (1 + 0.5 * z.clamp_max(0) + 2 * z.clamp_min(0)).float()
    torch.testing.assert_close(record["action_weights"], expected)
    assert metrics["ugrow/weight_nonunit_fraction"] > 0
    assert "dvac_v" not in record and record["policy_version"] == 3
    assert record["action"].eq(0.2).all()
    pool = SuccessReplay(42, str(tmp_path / "data"), signal_spec=dict(UGROW_SIGNAL_SPEC))
    pool.add_episodes(first + second)
    old = record["action_weights"].clone()
    m.annotate([], torch.tensor([1., -20., 400.]))
    assert torch.equal(record["action_weights"], old)
    batch = pool.sample(32)["forward_inputs"]
    loss_elements = torch.ones_like(batch["action_valid_mask"], dtype=torch.float32, requires_grad=True)
    loss = masked_fm_loss(loss_elements, batch["action_valid_mask"], batch["action_weights"])
    loss.backward()
    expected_grad = batch["action_weights"][..., None].expand_as(loss_elements) / (32 * 3 * 14)
    torch.testing.assert_close(loss_elements.grad, expected_grad)


def test_u_checkpoint_roundtrip_and_no_dv_u_mixing(tmp_path):
    episodes, moments = collect([[0.01, 0.1, 0.5], [0.02, 0.2, 0.8]])
    m = mapper()
    m.annotate(episodes, moments)
    pool = SuccessReplay(42, str(tmp_path / "data"), signal_spec=dict(UGROW_SIGNAL_SPEC))
    pool.add_episodes(episodes)
    pool.save_checkpoint(tmp_path / "cp")
    expected = pool.sample(4)
    restored = SuccessReplay(3, str(tmp_path / "new"), signal_spec=dict(UGROW_SIGNAL_SPEC))
    restored.load_checkpoint(tmp_path / "cp")
    for key, tensor in restored.sample(4)["forward_inputs"].items():
        assert torch.equal(tensor, expected["forward_inputs"][key])
    m2 = mapper()
    m2.load_state_dict(m.state_dict())
    assert m2.round_id == m.round_id
    with pytest.raises(ValueError):
        OnlineBCDvac().load_state_dict(m.state_dict())
    with pytest.raises(ValueError):
        SuccessReplay(3, str(tmp_path / "old")).load_checkpoint(tmp_path / "cp")
    bad = copy.deepcopy(episodes)
    bad[0][0]["ugrow_signal_version"] = torch.tensor(2)
    with pytest.raises(ValueError):
        restored.add_episodes(bad)
    state = torch.load(tmp_path / "cp/success_replay.pt", weights_only=True)
    state["signal_spec"]["epsilon"] = 1e-6
    torch.save(state, tmp_path / "cp/success_replay.pt")
    with pytest.raises(ValueError):
        restored.load_checkpoint(tmp_path / "cp")


def test_actor_rejects_old_checkpoint_before_model_load(tmp_path):
    from rlinf.workers.actor.fsdp_online_bc_policy_worker import EmbodiedOnlineBCFSDPPolicy

    calls = []
    actor = SimpleNamespace(
        ugrow_enabled=True, _rank=0, dvac=mapper(),
        _strategy=SimpleNamespace(load_checkpoint=lambda **kw: calls.append("load")),
    )
    with pytest.raises(ValueError, match="distinct U checkpoint"):
        EmbodiedOnlineBCFSDPPolicy.load_checkpoint(actor, tmp_path)
    assert calls == []


def test_actor_u_checkpoint_restores_mapper_replay_rng_and_learner(tmp_path):
    from rlinf.workers.actor.fsdp_online_bc_policy_worker import EmbodiedOnlineBCFSDPPolicy

    episodes, moments = collect([[0.01, 0.1, 0.5], [0.02, 0.2, 0.8]])
    m = mapper()
    m.annotate(episodes, moments)
    pool = SuccessReplay(42, str(tmp_path / "data"), signal_spec=dict(UGROW_SIGNAL_SPEC))
    pool.add_episodes(episodes)
    calls = []
    actor = SimpleNamespace(
        ugrow_enabled=True, _rank=0, dvac=m, replay_buffer=pool,
        update_step=5, model=None, optimizer=None, lr_scheduler=None,
        checkpoint_format="local_shard", is_weight_offloaded=False,
        is_optimizer_offloaded=False, device=torch.device("cpu"),
        load_param_and_grad=lambda device: calls.append("parameters"),
        load_optimizer=lambda device: calls.append("optimizer"),
        _strategy=SimpleNamespace(
            save_checkpoint=lambda **kw: calls.append("save"),
            load_checkpoint=lambda **kw: calls.append("load"),
        ),
    )
    EmbodiedOnlineBCFSDPPolicy.save_checkpoint(actor, tmp_path / "cp", 1)
    assert (tmp_path / "cp/online_bc/rank_0/ugrow.pt").is_file()
    assert not (tmp_path / "cp/online_bc/rank_0/dvac.pt").exists()
    expected = pool.sample(6)
    actor.dvac = mapper()
    actor.replay_buffer = SuccessReplay(0, str(tmp_path / "new"), signal_spec=dict(UGROW_SIGNAL_SPEC))
    actor.update_step = 0
    actor.is_weight_offloaded = actor.is_optimizer_offloaded = True
    EmbodiedOnlineBCFSDPPolicy.load_checkpoint(actor, tmp_path / "cp")
    assert actor.update_step == 5 and actor.dvac.round_id == 1
    assert calls == ["save", "parameters", "optimizer", "load"]
    for key, value in actor.replay_buffer.sample(6)["forward_inputs"].items():
        assert torch.equal(value, expected["forward_inputs"][key])
    actor.ugrow_enabled = False
    with pytest.raises(ValueError, match="Cannot load a U checkpoint"):
        EmbodiedOnlineBCFSDPPolicy.load_checkpoint(actor, tmp_path / "cp")
    assert calls == ["save", "parameters", "optimizer", "load"]


def test_rollout_u_only_for_training_and_retains_ode_mode():
    from contextlib import nullcontext
    from inspect import unwrap
    from omegaconf import OmegaConf
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    seen = []
    def predict_action_batch(**kwargs):
        seen.append(kwargs)
        return torch.zeros(2, 50, 14), {"forward_inputs": {}}
    worker = SimpleNamespace(
        algorithm_cfg=OmegaConf.create({"loss_type": "online_bc", "online_bc": {
            "dvac": {"enabled": True, "signal_kind": "ugrow_10_5", "tail_steps": 0}}}),
        model_cfg=SimpleNamespace(model_type="openpi"), enable_dagger=False,
        expert_model=None, hf_model=SimpleNamespace(predict_action_batch=predict_action_batch),
        _train_sampling_params={}, _eval_sampling_params={}, worker_timer=lambda *a, **k: nullcontext(),
    )
    predict = unwrap(MultiStepRolloutWorker.predict)
    predict(worker, {}, mode="train")
    predict(worker, {}, mode="eval")
    assert seen[0]["mode"] == seen[1]["mode"] == "eval"
    assert seen[0]["ugrow_enabled"] and "ugrow_enabled" not in seen[1]
    assert "dvac_tail_steps" not in seen[0]
