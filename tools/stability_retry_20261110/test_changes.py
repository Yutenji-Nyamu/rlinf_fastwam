# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""CPU acceptance of changed loss and immutable reference replay/resume."""

import contextlib
import json
import math
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from omegaconf import OmegaConf
from prepare import BASE, EXPORT, save

torch.set_num_threads(2)
torch.manual_seed(10)
method = sys.argv[1]
sys.path.insert(0, str(EXPORT / method))
if method == "dsrl":
    from rlinf.algorithms.noise_prior import gaussian_prior_kl
    from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy

    z = torch.randn(32, 1600)
    lp = -0.5 * (z.square() + math.log(2 * math.pi)).sum(-1)
    assert gaussian_prior_kl(z, lp).abs().max() < 1e-6
    mu = torch.full((1600,), 0.8, requires_grad=True)
    eps = torch.randn(1024, 1600)
    z = mu + eps
    lp = -0.5 * (eps.square() + math.log(2 * math.pi)).sum(-1)
    kl = gaussian_prior_kl(z, lp).mean()
    kl.backward()
    assert abs(kl.item() - 0.32) < 0.005 and torch.all(mu.grad > 0)
    m = GaussianPolicy(128, 1600, low=-2, high=2)
    m.probability_float32 = True
    z, lp = m(torch.randn(64, 128), sample=True)
    loss = gaussian_prior_kl(z, lp).mean()
    loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in m.parameters())
    result = {
        "analytic_gaussian_zero": True,
        "shift_kl": float(kl),
        "gradient_restores_mean": True,
        "bounded_policy_finite_gradients": True,
    }
else:
    import rlinf.workers.actor.fsdp_rlt_ac_policy_worker as w
    from rlinf.data.schema.embodied_types import Trajectory
    from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
    from rlinf.data.storage.replay.dataset import ReplayBufferDataset
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

    c = OmegaConf.load(BASE / "dojo-rlt-clean-g6/config/formal.yaml")
    m = RLTMLPPolicy(
        z_dim=2048,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=10,
        output_activation="identity",
    )
    a = object.__new__(w.RLTACFSDPPolicy)
    a.cfg = c
    a.update_step = 100000
    a._metric = lambda x: float(x.detach())
    a.worker_timer = lambda *args, **kw: contextlib.nullcontext()

    def forward(forward_type, obs, **kw):
        if forward_type == w.ForwardType.SAC:
            return m.sac_forward(obs, **kw)
        return m.sac_q_forward(obs, **kw)

    a.model = forward
    obs = {
        "z_rl": torch.randn(32, 2048),
        "proprio": torch.randn(32, 14),
        "ref_chunk": torch.randn(32, 10, 14),
    }
    torch.manual_seed(42)
    loss, _, metrics = w.RLTACLossMixin.forward_actor.__wrapped__(
        a, {"curr_obs": obs, "actions": obs["ref_chunk"]}
    )
    torch.manual_seed(42)
    pred = m.sac_forward(obs)[0]
    expected = (pred.flatten(1) - obs["ref_chunk"].flatten(1)).square().mean()
    assert (
        torch.allclose(loss, expected)
        and metrics["q_weight"] == 0
        and metrics["reference_dropout_prob"] == 0
    )
    loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in m.parameters())
    nxt, _, _ = a._next_actions_for_critic_target(obs)
    assert torch.equal(nxt, obs["ref_chunk"].flatten(1))

    def trajectory(value):
        t = Trajectory(max_episode_length=10)
        t.actions = torch.full((1, 1, 140), float(value))
        t.rewards = torch.zeros(1, 1, 10)
        t.dones = t.terminations = t.truncations = torch.zeros(
            1, 1, 10, dtype=torch.bool
        )
        t.curr_obs = {"x": torch.full((1, 1, 2), float(value))}
        t.next_obs = {"x": torch.full((1, 1, 2), float(value + 1))}
        t.model_weights_id = "cpu-test"
        return t

    def buf():
        return TrajectoryReplayBuffer(
            seed=1,
            enable_cache=True,
            cache_size=4,
            sample_window_size=4,
            auto_save=False,
        )

    a.replay_buffer = buf()
    a.demo_buffer = buf()
    a.cfg.algorithm.rlt_reference_replay_size = 4
    a.version = 0
    a._transition_replay_trajectories = lambda t: ([t], 0)
    a._accumulate_rlt_dvac_baseline = lambda _: None
    w.collect_trajectory_replay_metrics = lambda *x, **kw: {}
    a._ingest_rollout_trajectories([trajectory(1) for _ in range(4)])
    a._ingest_rollout_trajectories([trajectory(9) for _ in range(8)])
    assert (
        a.demo_buffer.total_samples == 4
        and a.demo_buffer.sample(4)["actions"].eq(1).all()
    )
    with tempfile.TemporaryDirectory(prefix="rlt-reference-test-") as tmp:
        a.demo_buffer.save_checkpoint(tmp)
        restored = buf()
        restored.load_checkpoint(tmp)
        assert restored.total_samples == 4 and restored.sample(4)["actions"].eq(1).all()
        ds = ReplayBufferDataset(a.replay_buffer, restored, 8, 1, 1)
        batch = next(iter(ds))["actions"]
        assert batch.shape[0] == 8
        assert (batch[:, 0] == 1).sum() == 4 and (batch[:, 0] == 9).sum() == 4
    result = {
        "actual_actor_loss_is_pure_bc": True,
        "reference_bootstrap": True,
        "reference_survives_online_overwrite": True,
        "reference_roundtrip": True,
        "half_reference_half_online": True,
    }
save(BASE / f"test-changes-{method}.json", result)
print(json.dumps(result))
