"""Focused CPU checks for the Fast-WAM/current-RLT interface."""

import json

import torch

from rlinf.models.embodiment.fastwam.rlt_policy import (
    FastWAMRLTPolicy, feature_identity,
)
from rlinf.models.embodiment.fastwam.rlt_stage1_data import lerobot_frame_to_env_obs
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def test_output_activation_preserves_pi0_and_allows_fast_targets():
    obs = {"z_rl": torch.zeros(1, 8), "proprio": torch.zeros(1, 14),
           "ref_chunk": torch.zeros(1, 24, 14)}
    for activation, expected in [("tanh", torch.tanh(torch.tensor(1.3))), ("identity", 1.3)]:
        model = RLTMLPPolicy(8, 14, 14, 24, output_activation=activation)
        for parameter in model.parameters():
            parameter.data.zero_()
        model.actor_mean.bias.data.fill_(1.3)
        actual, _, _ = model.sac_forward(obs, deterministic=True)
        torch.testing.assert_close(actual, torch.full_like(actual, float(expected)))
        actual.sum().backward()
        assert model.actor_mean.bias.grad.abs().min() > 0
    assert RLTMLPPolicy(8, 14, 14, 24).output_activation == "tanh"


def test_decode_replaces_only_committed_24(monkeypatch):
    policy = object.__new__(FastWAMRLTPolicy)
    torch.nn.Module.__init__(policy)
    policy.processor = object()
    template = torch.arange(32 * 14).reshape(1, 32, 14).float()
    before = template.clone()
    seen = []
    def decode(value, processor):
        seen.append(value.clone())
        return value * 2 + 3
    monkeypatch.setattr("rlinf.models.embodiment.fastwam.rlt_policy.denormalize_actions", decode)
    actions = torch.full((1, 24, 14), 1.3)
    result = policy.decode_rlt_action(actions, {"teacher_template": template})
    torch.testing.assert_close(seen[0][:, :24], actions)
    torch.testing.assert_close(seen[0][:, 24:], before[:, 24:])
    torch.testing.assert_close(template, before)
    torch.testing.assert_close(result, actions * 2 + 3)


def test_lerobot_raw_cameras_and_state():
    frame = {"observation.state": torch.arange(14).float(), "task": "adjust bottle"}
    for key, value in [("cam_high", .2), ("cam_left_wrist", .4), ("cam_right_wrist", .8)]:
        frame[f"observation.images.{key}"] = torch.full((3, 4, 5), value)
    obs = lerobot_frame_to_env_obs(frame)
    assert obs["main_images"].shape == (1, 4, 5, 3)
    assert obs["main_images"].dtype == torch.uint8
    assert obs["wrist_images"][0, 0, 0, 0, 0] == 102
    assert obs["wrist_images"][0, 1, 0, 0, 0] == 204
    torch.testing.assert_close(obs["states"][0], frame["observation.state"])


def test_identity_tracks_stats_and_output_semantics(tmp_path):
    teacher, stats = tmp_path / "teacher.pt", tmp_path / "stats.json"
    teacher.write_bytes(b"pinned-test-artifact")
    stats.write_text(json.dumps({"mean": 0}))
    first = feature_identity(teacher, stats, {"input_dim": 3072})
    stats.write_text(json.dumps({"mean": 1}))
    second = feature_identity(teacher, stats, {"input_dim": 3072})
    assert first != second
    assert first["student_output_activation"] == "identity"
    assert first["action_chunk"] == 24
