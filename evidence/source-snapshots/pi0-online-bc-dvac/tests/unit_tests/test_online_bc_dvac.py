# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from inspect import unwrap
from types import SimpleNamespace

import pytest
import torch

from rlinf.algorithms.online_bc_dvac import OnlineBCDvac, endpoint_variance, log_moments
from rlinf.data.online_bc import SuccessEpisodeCollector, SuccessReplay, masked_fm_loss


def row(v):
    return {
        "dvac_v": torch.as_tensor(v, dtype=torch.float32),
        "action_valid_mask": torch.ones(len(v), 2, dtype=torch.bool),
    }


def test_endpoint_population_variance():
    x = torch.arange(2 * 3 * 50 * 14.0).reshape(2, 3, 50, 14).requires_grad_()
    expected = ((x - x.mean(1, keepdim=True)).square().mean(1)).sum(-1)
    v = endpoint_variance(x)
    torch.testing.assert_close(v, expected)
    assert v.shape == (2, 50) and not v.requires_grad
    with pytest.raises(ValueError):
        endpoint_variance(torch.ones(2, 1, 50, 14))


def test_unit_weights_exact_loss_gradient_and_detachment():
    a = torch.randn(2, 50, 14, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    mask = torch.ones_like(a, dtype=torch.bool)
    mask[:, -2:] = False
    w = torch.ones(2, 50, requires_grad=True)
    unweighted = masked_fm_loss(a.square(), mask)
    weighted = masked_fm_loss(b.square(), mask, w)
    assert torch.equal(unweighted, weighted)
    unweighted.backward()
    weighted.backward()
    assert torch.equal(a.grad, b.grad) and w.grad is None
    with pytest.raises(ValueError):
        masked_fm_loss(a, mask, torch.ones(2))


def test_weights_act_before_horizon_reduction():
    values = torch.tensor([[[1.0], [3.0]]], requires_grad=True)
    mask = torch.ones_like(values, dtype=torch.bool)
    loss = masked_fm_loss(values, mask, torch.tensor([[0.5, 1.5]]))
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    torch.testing.assert_close(values.grad.flatten(), torch.tensor([0.25, 0.75]))


def test_collector_counts_failed_once_excludes_post_terminal_and_archives_success(
    tmp_path,
):
    c = SuccessEpisodeCollector(2, dvac_log_eps=1e-12)
    obs = {
        "observation/state": torch.zeros(2, 14),
        "dvac_v": torch.arange(1.0, 101).reshape(2, 50),
    }
    c.append(obs, torch.ones(2, 50, 14), [True, False], torch.ones(2, 50))
    c.append(obs, torch.ones(2, 50, 14), [True, True], torch.ones(2, 50))
    moments = c.drain_dvac_moments()
    torch.testing.assert_close(moments, log_moments(obs["dvac_v"], 1e-12))
    assert moments[0] == 100 and c.drain_dvac_moments().eq(0).all()
    episodes = c.drain()
    assert len(episodes) == 1 and len(episodes[0]) == 1
    dvac = OnlineBCDvac()
    dvac.annotate(episodes, moments)
    pool = SuccessReplay(1, str(tmp_path / "data"))
    pool.add_episodes(episodes)
    saved = torch.load(tmp_path / "data/batch_000000.pt", weights_only=True)
    assert saved[0][0]["action_weights"].eq(1).all()
    torch.testing.assert_close(saved[0][0]["dvac_v"], obs["dvac_v"][0])
    assert "action_weights" in pool.sample(2)["forward_inputs"]


def test_calibrate_from_past_rounds_freeze_weights_bounds_and_restore(tmp_path):
    d = OnlineBCDvac(window=2)
    first = [[row([0.01, 0.1, 1.0])]]
    moments = log_moments(first[0][0]["dvac_v"], 1e-12)
    assert d.annotate(first, moments)["dvac/calibrated"] == 0
    assert first[0][0]["action_weights"].eq(1).all()
    second = [[row([0.001, 0.1, 10.0])]]
    metrics = d.annotate(second, moments * 2)
    assert metrics["dvac/reference_positions"] == 3
    w = second[0][0]["action_weights"]
    assert 0 <= w.min() < 1 < w.max() <= 2
    torch.testing.assert_close(w.mean(), torch.tensor(1.0))
    assert first[0][0]["action_weights"].eq(1).all()
    torch.save(d.state_dict(), tmp_path / "dvac.pt")
    restored = OnlineBCDvac(window=2)
    restored.load_state_dict(torch.load(tmp_path / "dvac.pt", weights_only=True))
    third, fourth = [[row([0.01, 1.0, 100.0])]], [[row([0.01, 1.0, 100.0])]]
    m1, m2 = d.annotate(third, moments), restored.annotate(fourth, moments)
    assert m1 == m2 and len(d.history) == 2
    assert torch.equal(third[0][0]["action_weights"], fourth[0][0]["action_weights"])
    with pytest.raises(ValueError):
        OnlineBCDvac(alpha=0.5)
    with pytest.raises(ValueError):
        OnlineBCDvac().load_state_dict(d.state_dict())


def test_weight_centering_respects_mask_and_empty_round():
    d = OnlineBCDvac()
    moments = log_moments(torch.tensor([0.01, 0.1, 1.0]), 1e-12)
    d.annotate([], moments)
    r = row([0.01, 0.1, 1.0])
    r["action_valid_mask"][0] = False
    r["action_valid_mask"][1, 1] = False
    d.annotate([[r]], torch.zeros(3))
    q = r["action_valid_mask"].sum(-1)
    torch.testing.assert_close(
        (r["action_weights"] * q).sum() / q.sum(), torch.tensor(1.0)
    )


def test_real_sampler_recording_preserves_actions_rng_and_forward_count():
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0ForRLActionPrediction,
    )

    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.action_in_proj = torch.nn.Linear(32, 2)
    model.config = SimpleNamespace(
        num_steps=4,
        action_horizon=50,
        action_dim=32,
        action_chunk=50,
        action_env_dim=14,
        joint_logprob=False,
        is_nft=False,
    )
    model.use_vlm_value = False
    model.sample_noise = lambda shape, device: torch.randn(shape, device=device)
    model.get_logprob_norm = lambda x, mean, std: torch.zeros_like(x)
    model._init_nft_state = lambda *args: {}
    model._update_nft_state = lambda *args: None
    previews = []

    def velocity(x, idx, *args):
        v = x * 0.2 + idx * 0.1
        v[:, :, 14:] = 999  # Padding must not enter DVAC.
        previews.append(x[:, :, :14] - (1 - idx / 4) * v[:, :, :14])
        return x - 0.25 * v, torch.zeros_like(x), torch.zeros(x.shape[0]), v

    model.sample_mean_var_val = velocity
    state, noise = torch.zeros(2, 32), torch.ones(2, 50, 32)
    outputs, rngs = [], []
    for tail in (0, 3):
        previews.clear()
        torch.manual_seed(123)
        outputs.append(
            model._sample_actions_with_prefix_cache(
                state, None, None, None, noise=noise, mode="eval", dvac_tail_steps=tail
            )
        )
        rngs.append(torch.get_rng_state().clone())
        assert len(previews) == 4
    for key in outputs[0]:
        assert torch.equal(outputs[0][key], outputs[1][key]), key
    assert torch.equal(*rngs)
    torch.testing.assert_close(
        outputs[1]["dvac_v"], endpoint_variance(torch.stack(previews[-3:], 1))
    )


def test_sft_weight_reaches_native_unreduced_loss(monkeypatch):
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0ForRLActionPrediction,
    )

    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter("probe", torch.nn.Parameter(torch.tensor(1.0)))
    model.config = SimpleNamespace(use_rlt=False, action_chunk=50, action_env_dim=14)
    model.gradient_checkpointing_disable = lambda: None
    obs = Observation(
        images={},
        image_masks={},
        state=torch.zeros(1, 32),
        tokenized_prompt=torch.zeros(1, 4, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    actions = torch.arange(50.0).reshape(1, 50, 1).expand(1, 50, 32)
    monkeypatch.setattr(
        PI0Pytorch, "forward", lambda self, obs, a: a.square() * self.probe
    )
    weights = torch.linspace(0.5, 1.5, 50).unsqueeze(0)
    loss = model.sft_forward(
        (obs, actions),
        use_action_chunk_loss=True,
        action_valid_mask=torch.ones(1, 50, 14, dtype=torch.bool),
        action_weights=weights,
    )
    expected = (actions[:, :, :14].square() * weights[..., None]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(model.probe.grad, expected)


def test_rollout_records_only_training_but_keeps_bc_ode_mode():
    from omegaconf import OmegaConf

    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    seen = []

    class Model:
        def predict_action_batch(self, **kwargs):
            seen.append(kwargs)
            return torch.zeros(2, 50, 14), {"forward_inputs": {}}

    worker = SimpleNamespace(
        algorithm_cfg=OmegaConf.create(
            {
                "loss_type": "online_bc",
                "online_bc": {"dvac": {"enabled": True, "tail_steps": 3}},
            }
        ),
        model_cfg=SimpleNamespace(model_type="openpi"),
        enable_dagger=False,
        expert_model=None,
        hf_model=Model(),
        _train_sampling_params={},
        _eval_sampling_params={},
        worker_timer=lambda *a, **k: nullcontext(),
    )
    predict = unwrap(MultiStepRolloutWorker.predict)
    predict(worker, {}, mode="train")
    predict(worker, {}, mode="eval")
    assert seen[0]["mode"] == seen[1]["mode"] == "eval"
    assert seen[0]["dvac_tail_steps"] == 3 and "dvac_tail_steps" not in seen[1]
    worker.algorithm_cfg.online_bc.dvac.enabled = False
    predict(worker, {}, mode="train")
    assert "dvac_tail_steps" not in seen[2]


def test_actor_checkpoint_keeps_dvac_with_existing_replay_and_learner(tmp_path):
    from rlinf.workers.actor.fsdp_online_bc_policy_worker import (
        EmbodiedOnlineBCFSDPPolicy,
    )

    calls = []
    strategy = SimpleNamespace(
        save_checkpoint=lambda **kw: calls.append("save"),
        load_checkpoint=lambda **kw: calls.append("load"),
    )
    pool = SuccessReplay(3, str(tmp_path / "data"))
    d = OnlineBCDvac()
    episodes = [[row([0.01, 0.1, 1])]]
    d.annotate(episodes, log_moments(episodes[0][0]["dvac_v"], 1e-12))
    pool.add_episodes(episodes)
    actor = SimpleNamespace(
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        _strategy=strategy,
        model=None,
        optimizer=None,
        lr_scheduler=None,
        checkpoint_format="local_shard",
        _rank=0,
        replay_buffer=pool,
        update_step=10,
        dvac=d,
    )
    path = tmp_path / "checkpoint"
    EmbodiedOnlineBCFSDPPolicy.save_checkpoint(actor, path, 1)
    actor.update_step = 99
    actor.dvac = OnlineBCDvac()
    actor.replay_buffer = SuccessReplay(33, str(tmp_path / "data2"))
    EmbodiedOnlineBCFSDPPolicy.load_checkpoint(actor, path)
    assert calls == ["save", "load"] and actor.update_step == 10
    assert actor.dvac.round_id == 1 and len(actor.replay_buffer) == 1
    assert actor.replay_buffer.records[0]["action_weights"].eq(1).all()


def test_dvac_config_group_extends_existing_primary_without_replacing_recipe():
    import os
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        cfg = compose(
            config_name="robotwin_adjust_bottle_online_bc_openpi",
            overrides=["+bc_dvac=default"],
        )
    assert cfg.cluster.component_placement["actor,env,rollout"] == "7"
    assert (
        cfg.algorithm.online_bc.dvac.enabled
        and cfg.algorithm.online_bc.dvac.tail_steps == 3
    )
    assert cfg.env.train.total_num_envs == 32 and cfg.env.train.rollout_epoch == 1
    assert cfg.actor.micro_batch_size == 32 and cfg.actor.global_batch_size == 1024
    assert cfg.algorithm.update_epoch == 10 and cfg.actor.model.num_steps == 4
