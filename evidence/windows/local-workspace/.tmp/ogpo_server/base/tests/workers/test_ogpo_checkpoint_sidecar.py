from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.ogpo_replay import OGPOPrimitiveRow, OGPOReplayBuffer
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.actor.fsdp_ogpo_policy_worker import EmbodiedOGPOFSDPPolicy


class _FakeFlowModel:
    def __init__(self):
        self.shadow = {"0:weight": torch.tensor([1.25], dtype=torch.float32)}

    def __call__(self, *, operation, **kwargs):
        if operation == "ema_shadow_state":
            return {name: value.clone() for name, value in self.shadow.items()}
        if operation == "load_ema_shadow_state":
            self.shadow = {
                name: value.clone() for name, value in kwargs["state"].items()
            }
            return None
        raise ValueError(operation)


def _row() -> OGPOPrimitiveRow:
    observation = {
        "main_images": torch.zeros(2, 2, 3, dtype=torch.uint8),
        "states": torch.zeros(14),
        "prompt_utf8": torch.tensor([97, 0], dtype=torch.uint8),
        "prompt_length": torch.tensor(1),
    }
    return OGPOPrimitiveRow(
        observation=observation,
        next_observation={name: value.clone() for name, value in observation.items()},
        action_model=torch.zeros(32),
        action=torch.zeros(14),
        reward=0.0,
        terminated=False,
        truncated=False,
        episode_id=0,
        step_id=0,
    )


def _worker() -> EmbodiedOGPOFSDPPolicy:
    worker = object.__new__(EmbodiedOGPOFSDPPolicy)
    worker._rank = 0
    worker._world_size = 1
    worker.cfg = OmegaConf.create(
        {"actor": {"seed": 1234}, "algorithm": {"gamma": 0.999}}
    )
    worker.ogpo_cfg = OmegaConf.create(
        {
            "source_fingerprint": "source",
            "norm_fingerprint": "norm",
            "model_horizon": 50,
            "execution_horizon": 10,
            "model_action_dim": 32,
            "active_action_dim": 14,
            "flow_steps": 4,
            "sigma_init": 0.01,
            "gaussian_clip": 3.0,
            "normalize_denoising_horizon": True,
            "normalize_act_space_dimension": True,
            "num_q_heads": 10,
            "critic_hidden_dims": [16, 16],
            "replay_capacity": 4,
            "state_batch_size": 1,
            "candidate_group_size": 2,
            "start_training_rows": 0,
            "total_online_rows": 4,
            "utd_q": 1.0,
            "utd_pi": 1.0,
            "clip_epsilon": 0.01,
            "bc_coeff": 1.0,
            "actor_tau": 0.005,
            "critic_tau": 0.05,
            "critic_lr": 3e-4,
            "critic_adam_beta1": 0.9,
            "critic_adam_beta2": 0.999,
            "critic_adam_eps": 1e-8,
            "critic_weight_decay": 1e-5,
            "critic_grad_clip": 1.0,
        }
    )
    worker.replay = OGPOReplayBuffer(
        capacity=4,
        max_sequence_length=10,
        action_dim=14,
        model_action_dim=32,
        seed=1234,
    )
    worker.replay.add(_row())
    worker.model = _FakeFlowModel()
    worker.critic = None
    worker.target_critic = None
    worker.critic_optimizer = None
    worker.critic_feature_dim = None
    worker.global_online_rows = 1
    worker.pending_actor_updates = 1.0
    worker.pending_critic_updates = 1.0
    worker.actor_updates = 2
    worker.critic_updates = 2
    worker.policy_version = 2
    worker._episode_counter = 1
    return worker


def test_critic_optimizer_uses_configured_weight_decay() -> None:
    worker = _worker()
    worker.device = torch.device("cpu")

    worker._init_critic(feature_dim=8)

    assert worker.critic_optimizer is not None
    assert worker.critic_optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        1e-5
    )


def test_sidecar_roundtrip_and_manifest_preflight(monkeypatch, tmp_path) -> None:
    base_save_calls = []
    base_load_calls = []
    monkeypatch.setattr(
        EmbodiedFSDPActor,
        "save_checkpoint",
        lambda self, path, step: base_save_calls.append((path, step)),
    )
    monkeypatch.setattr(
        EmbodiedFSDPActor,
        "load_checkpoint",
        lambda self, path: base_load_calls.append(path),
    )
    worker = _worker()
    checkpoint = tmp_path / "checkpoint"
    worker.save_checkpoint(str(checkpoint), step=7)

    manifest_path = checkpoint / "ogpo_components" / "complete.json"
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["complete"] is True
    assert len(first_manifest["snapshot_id"]) == 32
    assert base_save_calls == [(str(checkpoint), 7)]

    worker.replay = OGPOReplayBuffer(
        capacity=4,
        max_sequence_length=10,
        action_dim=14,
        model_action_dim=32,
        seed=1234,
    )
    worker.model.shadow = {}
    worker.global_online_rows = 0
    worker.policy_version = 0
    worker.load_checkpoint(str(checkpoint))
    assert base_load_calls == [str(checkpoint)]
    assert len(worker.replay) == 1
    assert worker.global_online_rows == 1
    assert worker.policy_version == 2
    torch.testing.assert_close(
        worker.model.shadow["0:weight"], torch.tensor([1.25])
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_id"] = "mixed-sidecar"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot_id"):
        worker.load_checkpoint(str(checkpoint))
    assert base_load_calls == [str(checkpoint)]


def test_failed_sidecar_write_leaves_no_completion_marker(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        EmbodiedFSDPActor, "save_checkpoint", lambda self, path, step: None
    )
    worker = _worker()
    checkpoint = tmp_path / "checkpoint"
    worker.save_checkpoint(str(checkpoint), step=7)
    manifest = checkpoint / "ogpo_components" / "complete.json"
    assert manifest.is_file()

    def fail_save(*_args, **_kwargs):
        raise OSError("fixture write failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="sidecar write"):
        worker.save_checkpoint(str(checkpoint), step=8)
    assert not manifest.exists()
