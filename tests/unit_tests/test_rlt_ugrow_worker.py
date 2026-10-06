"""U replay transport, production BC gradients and checkpoint contracts."""

import copy
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.transition import extract_rlt_obs_from_forward_inputs
from rlinf.algorithms.ugrow_signal import UGROW_SIGNAL_SPEC
from rlinf.data.storage.replay.buffer import TrajectoryCache
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    RLTACFSDPPolicy,
    RLTACLossMixin,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class _TinyActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_actions = torch.nn.Parameter(torch.linspace(0.1, 1.0, 140))
        self.critic_bias = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, *, forward_type, obs, actions=None, **_kwargs):
        if forward_type == ForwardType.SAC:
            pi = self.actor_actions.expand(obs["ref_chunk"].shape[0], -1)
            return pi, pi.new_zeros(pi.shape[0], 1), None
        if forward_type == ForwardType.SAC_Q:
            q = actions.mean(dim=-1, keepdim=True) * self.critic_bias
            return torch.cat((q, q - 0.1), dim=-1)
        raise AssertionError(forward_type)


@pytest.fixture
def make_worker(monkeypatch):
    def base_init(worker, cfg):
        worker.cfg = cfg
        worker._rank = 0
        worker._world_size = 1
        worker.update_step = 0
        worker.version = 0

    monkeypatch.setattr(EmbodiedSACFSDPPolicy, "__init__", base_init)

    def make(*, source="ugrow_10_5", mode="apply", dropout=0.0, spec=None):
        signal = dict(signal_source=source, signal_spec=copy.deepcopy(
            UGROW_SIGNAL_SPEC if spec is None else spec
        )) if source == "ugrow_10_5" else {}
        cfg = OmegaConf.create({
            "actor": {"global_batch_size": 512, "micro_batch_size": 256,
                      "model": {"num_action_chunks": 10, "action_dim": 14}},
            "rollout": {"rlt_feature_model": {"openpi": {
                "rlt_ugrow_enabled": source == "ugrow_10_5",
                "num_steps": 10, "action_env_dim": 14,
            }}},
            "algorithm": {
                "rlt_dvac": dict(
                    mode=mode, mapping="two_level_batch",
                    application="success_episode_bc", success_target="reference",
                    outer_scope="successful_batch", applied_horizon=10,
                    alpha_local=1.0, alpha_chunk=1.0, success_scale=1.0,
                    factor_mapping="exp_mean", temperature_local=2.5,
                    temperature_chunk=2.5,
                    chunk_dropout=dict(enabled=mode != "off", probability=dropout, seed=42),
                    alpha_schedule=dict(
                        enabled=mode != "off",
                        local=dict(start_step=1, end_step=500, end_alpha=0.0),
                        chunk=dict(start_step=1, end_step=500, end_alpha=0.0),
                    ), **signal,
                ),
                "bc_weight": 2.5, "q_weight": 0.45,
                "reference_dropout_prob": 0.0,
                "rlt_resume": {"enable": True, "contract": {"stage1_manifest_id": "same-stage1"}},
            },
            "env": {"train": {"auto_reset": False}},
        })
        worker = RLTACFSDPPolicy(cfg)
        worker.model = _TinyActorCritic()
        return worker

    return make


def _batch():
    profile = torch.linspace(0.01, 0.4, 50)
    signal = profile.expand(512, -1) * torch.linspace(0.2, 1, 512)[:, None]
    ref = torch.linspace(-0.6, 0.9, 512 * 140).reshape(512, 10, 14)
    return {"curr_obs": {
        "ref_chunk": ref,
        "teacher_ugrow_u": signal,
        # A deliberately unrelated DV field catches accidentally selecting DV.
        "teacher_dvac_v": torch.ones(512, 3, 50),
        "episode_success": (torch.arange(512) % 2 == 0)[:, None],
    }, "actions": torch.zeros(512, 140)}


def _actor(worker, batch):
    return inspect.unwrap(RLTACLossMixin.forward_actor)(worker, batch)


def test_u_reaches_transition_and_replay_cache_without_using_dv():
    forward = dict(z_rl=torch.zeros(4, 8), proprio=torch.zeros(4, 14),
                   ref_chunk=torch.zeros(4, 10, 14),
                   teacher_ugrow_u=torch.linspace(0, 1, 200).reshape(4, 50))
    obs = extract_rlt_obs_from_forward_inputs(forward)
    cache = TrajectoryCache(max_size=2)
    cache.put(1, {"curr_obs": obs, "actions": torch.zeros(4, 140)})
    forward["teacher_ugrow_u"].zero_()
    saved = cache.get(1)["curr_obs"]["teacher_ugrow_u"]
    torch.testing.assert_close(saved, torch.linspace(0, 1, 200).reshape(4, 50))
    assert "teacher_dvac_v" not in obs


def test_precollection_ingest_logs_new_u_before_any_update(make_worker):
    worker = make_worker()
    rows = [SimpleNamespace(curr_obs={
        "teacher_ugrow_u": torch.full((1, 1, 50), value),
        "episode_success": torch.tensor([[[success]]]),
    }) for value, success in ((0.1, True), (0.3, False))]
    metrics = worker._ugrow_ingest_metrics(rows)
    assert worker.update_step == 0
    assert metrics["rlt_ugrow/rollout_query_count"] == 2
    assert metrics["rlt_ugrow/rollout_success_query_count"] == 1
    assert metrics["rlt_ugrow/rollout_u_mean"] == pytest.approx(0.2)
    assert metrics["rlt_ugrow/rollout_u_nonzero_count"] == 20
    rows[0].curr_obs.pop("teacher_ugrow_u")
    with pytest.raises(ValueError, match="teacher_ugrow_u"):
        worker._ugrow_ingest_metrics(rows)


def test_full_batch_u_weights_microbatch_gradients_and_unchanged_q(make_worker):
    batch = _batch()
    results = []
    for splits in (1, 2, 4):
        worker = make_worker()
        prepared, metrics = worker._prepare_global_batch(batch, train_actor=True)
        weights = prepared["rlt_dvac_new_weights"]
        success = batch["curr_obs"]["episode_success"].flatten()
        assert torch.equal(weights[~success], torch.ones_like(weights[~success]))
        assert (weights[success, -1] > weights[success, 0]).all()
        assert metrics["rlt_ugrow/w_nonuniform_count"] > 0
        assert metrics["rlt_ugrow/success_query_count"] == 256
        assert "rlt_dvac_new_weights" not in batch
        assert not weights.requires_grad
        prepared["curr_obs"] = dict(prepared["curr_obs"])
        del prepared["curr_obs"]["teacher_ugrow_u"]
        total_loss = torch.tensor(0.0)
        for micro in split_dict_to_chunk(prepared, splits):
            loss, _, _ = _actor(worker, micro)
            (loss / splits).backward()
            total_loss += loss.detach() / splits
        results.append((total_loss, worker.model.actor_actions.grad.clone()))
    for loss, grad in results[1:]:
        torch.testing.assert_close(loss, results[0][0])
        torch.testing.assert_close(grad, results[0][1])
    clean = make_worker(source="dvac", mode="off")
    clean_loss, _, clean_metrics = _actor(clean, batch)
    clean_loss.backward()
    active = make_worker()
    prepared, _ = active._prepare_global_batch(batch, train_actor=True)
    _, _, active_metrics = _actor(active, prepared)
    assert active_metrics["weighted_q"] == clean_metrics["weighted_q"]
    assert not torch.allclose(results[0][1], clean.model.actor_actions.grad)
    untouched, diagnostics = active._prepare_global_batch(batch, train_actor=False)
    assert untouched is batch and diagnostics == {}


def test_annealed_endpoint_and_dropout_preserve_clean_bc(make_worker):
    batch = _batch()
    clean_loss, _, _ = _actor(make_worker(source="dvac", mode="off"), batch)
    for endpoint in (False, True):
        worker = make_worker(dropout=0.2 if endpoint else 1.0)
        worker.version = 499 if endpoint else 0
        prepared, _ = worker._prepare_global_batch(batch, train_actor=True)
        assert torch.equal(prepared["rlt_dvac_new_weights"], torch.ones(512, 10))
        loss, _, _ = _actor(worker, prepared)
        torch.testing.assert_close(loss, clean_loss)


def test_signal_and_dv_resume_contracts_are_distinct(make_worker):
    u = make_worker()
    dv = make_worker(source="dvac")
    state = u._rlt_state_payload(runner_step=10)
    u._validate_rlt_state(state)
    assert json.loads(state["rlt_resume_contract"])["rlt_ugrow_signal_spec"] == UGROW_SIGNAL_SPEC
    with pytest.raises(ValueError, match="rlt_resume_contract"):
        dv._validate_rlt_state(state)
    with pytest.raises(ValueError, match="rlt_resume_contract"):
        u._validate_rlt_state(dv._rlt_state_payload(runner_step=10))
    bad_spec = dict(UGROW_SIGNAL_SPEC, epsilon=1e-6)
    with pytest.raises(ValueError, match="signal_spec"):
        make_worker(spec=bad_spec)


@pytest.mark.parametrize("kind", ["missing", "nan", "partial", "wrong_shape"])
def test_missing_or_invalid_u_never_falls_back_to_dv(make_worker, kind):
    batch = _batch()
    if kind == "missing":
        del batch["curr_obs"]["teacher_ugrow_u"]
    elif kind == "nan":
        batch["curr_obs"]["teacher_ugrow_u"][0, 0] = float("nan")
    elif kind == "partial":
        batch = split_dict_to_chunk(batch, 2)[0]
    else:
        batch["curr_obs"]["teacher_ugrow_u"] = torch.zeros(512, 3, 50)
    with pytest.raises(ValueError):
        make_worker()._prepare_global_batch(batch, train_actor=True)
