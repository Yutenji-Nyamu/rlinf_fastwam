# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
import torch

from rlinf.data.online_bc import SuccessEpisodeCollector, SuccessReplay, masked_fm_loss


def inputs(value, n=2):
    return {
        "observation/state": torch.full((n, 14), value),
        "tokenized_prompt": torch.zeros(n, 4, dtype=torch.int64),
        "action": torch.zeros(n, 50 * 14),
        "model_action": torch.ones(n, 50 * 32),
        "chains": torch.ones(n, 11, 50, 32),
    }


def test_success_keeps_pre_query_submitted_command_and_drops_post_terminal():
    collector = SuccessEpisodeCollector(2)
    commands = torch.ones(2, 50, 14)
    before = inputs(3)
    collector.append(before, commands, [False, False], torch.zeros(2, 50))
    before["observation/state"].zero_()
    commands.zero_()
    collector.append(
        inputs(4), torch.full((2, 50, 14), 2), [True, False], torch.ones(2, 50)
    )
    collector.append(inputs(99), commands, [True, False], torch.ones(2, 50))
    episodes = collector.drain()
    assert len(episodes) == 1 and len(episodes[0]) == 2
    row = episodes[0][0]
    assert row["observation/state"].eq(3).all()
    assert row["action"].eq(1).all()
    assert row["action_valid_mask"].shape == (50, 14)
    assert "model_action" not in row and "chains" not in row
    assert collector.drain() == []


def test_success_is_separate_from_termination_and_reset_preserves_completed():
    collector = SuccessEpisodeCollector(2)
    collector.append(
        inputs(1), torch.ones(2, 50, 14), [True, False], torch.zeros(2, 50)
    )
    collector.reset()
    collector.append(inputs(2), torch.ones(2, 50, 14), [False, True], torch.ones(2, 50))
    episodes = collector.drain()
    assert len(episodes) == 2
    assert all(len(episode) == 1 for episode in episodes)


def test_replay_cumulative_archives_and_checkpoint_restore_rng(tmp_path):
    collector = SuccessEpisodeCollector(2)
    collector.append(inputs(3), torch.ones(2, 50, 14), [True, True], torch.ones(2, 50))
    pool = SuccessReplay(1, str(tmp_path / "archive"))
    assert not pool.is_ready()
    pool.add_episodes([])
    pool.add_episodes(collector.drain())
    pool.save_checkpoint(tmp_path / "checkpoint")
    expected = pool.sample(6)
    restored = SuccessReplay(99, str(tmp_path / "new_archive"))
    restored.load_checkpoint(tmp_path / "checkpoint")
    actual = restored.sample(6)
    for key in expected["forward_inputs"]:
        torch.testing.assert_close(
            actual["forward_inputs"][key], expected["forward_inputs"][key]
        )
    assert restored.get_stats() == {"success_episodes": 2, "query_records": 2}
    assert (tmp_path / "archive/batch_000000.pt").is_file()


def test_masked_fm_loss_and_gradient():
    loss = torch.arange(12.0).reshape(2, 3, 2).requires_grad_()
    ones = torch.ones_like(loss, dtype=torch.bool)
    torch.testing.assert_close(masked_fm_loss(loss, ones), loss.mean())
    mask = ones.clone()
    mask[:, 2] = False
    result = masked_fm_loss(loss, mask)
    torch.testing.assert_close(result, loss[:, :2].mean())
    result.backward()
    assert loss.grad[:, 2].eq(0).all()


def test_empty_or_misaligned_mask_rejected():
    import pytest

    loss = torch.ones(2, 3, 2)
    for mask in (torch.zeros_like(loss), torch.ones(2, 3)):
        with pytest.raises(ValueError):
            masked_fm_loss(loss, mask)


def test_demo_weight_uses_native_sft_and_zero_does_not_read_demos():
    from contextlib import nullcontext
    from inspect import unwrap
    from types import SimpleNamespace

    from rlinf.workers.actor.fsdp_online_bc_policy_worker import (
        EmbodiedOnlineBCFSDPPolicy,
    )

    class Model:
        def prepare_dagger_sft_batch(self, batch):
            return "online"

        def __call__(self, *, data, **kwargs):
            return torch.tensor(2.0 if data == "online" else 6.0, requires_grad=True)

    actor = SimpleNamespace(
        model=Model(), demo_weight=0, worker_timer=lambda *a, **kw: nullcontext()
    )
    forward = unwrap(EmbodiedOnlineBCFSDPPolicy.forward_actor)
    batch = {"action_valid_mask": torch.ones(1, 50, 14, dtype=torch.bool)}
    torch.testing.assert_close(forward(actor, batch), torch.tensor(2.0))
    actor.demo_weight = 1.0
    actor.demo_iterator = iter(["demo"])
    torch.testing.assert_close(forward(actor, batch), torch.tensor(4.0))


def test_online_bc_config_is_teacher_free_expert_only_and_single_gpu():
    import os
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin_adjust_bottle_online_bc_openpi")
    assert cfg.algorithm.loss_type == "online_bc"
    assert cfg.rollout.expert_model is None
    assert cfg.actor.model.openpi.train_expert_only
    assert not cfg.actor.model.openpi.image_augmentation
    assert cfg.actor.model.precision is None
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype is None
    assert not cfg.actor.fsdp_config.use_orig_params
    assert cfg.actor.fsdp_config.checkpoint_format == "local_shard"
    assert (
        "GemmaDecoderLayer"
        not in cfg.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap
    )
    assert {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(
        cfg.actor.fsdp_config.wrap_policy.no_split_names
    )
    assert cfg.cluster.component_placement["actor,env,rollout"] == "6"
    assert cfg.actor.model.num_steps == cfg.actor.model.openpi.num_steps == 4
    assert cfg.env.train.total_num_envs == cfg.env.eval.total_num_envs == 32
    assert cfg.env.train.rollout_epoch == 1
    assert cfg.actor.micro_batch_size == 32
    assert cfg.actor.global_batch_size == 1024
    assert cfg.algorithm.online_bc.demo_weight == 0


def test_sft_preserves_native_float32_augmentation_inputs(monkeypatch):
    from types import SimpleNamespace

    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models_pytorch.preprocessing_pytorch import (
        IMAGE_KEYS,
        preprocess_observation_pytorch,
    )

    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0ForRLActionPrediction,
    )

    # Native OpenPI SFT keeps augmentation and projection inputs FP32.
    # The config test above prevents FSDP from globally overriding that dtype.
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter("probe", torch.nn.Parameter(torch.zeros(1)))
    model.config = SimpleNamespace(use_rlt=False, action_chunk=50, action_env_dim=14)
    model.gradient_checkpointing_disable = lambda: None
    masks = {k: torch.ones(2, dtype=torch.bool) for k in IMAGE_KEYS}
    obs = Observation(
        images={
            k: torch.zeros(2, 3, 224, 224, dtype=torch.float32) for k in IMAGE_KEYS
        },
        image_masks=masks,
        state=torch.zeros(2, 32, dtype=torch.float32),
        tokenized_prompt=torch.zeros(2, 8, dtype=torch.int64),
        tokenized_prompt_mask=torch.ones(2, 8, dtype=torch.bool),
    )

    def native_forward(self, observation, actions):
        assert all(v.dtype == torch.float32 for v in observation.images.values())
        assert observation.tokenized_prompt.dtype == torch.int64
        assert all(v.dtype == torch.bool for v in observation.image_masks.values())
        torch.manual_seed(42)
        augmented = preprocess_observation_pytorch(observation, train=True)
        assert all(torch.isfinite(v).all() for v in augmented.images.values())
        assert actions.dtype == torch.float32
        return actions.square() + self.probe

    monkeypatch.setattr(PI0Pytorch, "forward", native_forward)
    loss = model.sft_forward(
        (obs, torch.ones(2, 50, 32, dtype=torch.bfloat16)),
        use_action_chunk_loss=True,
        action_valid_mask=torch.ones(2, 50, 14, dtype=torch.bool),
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(model.probe.grad, torch.ones(1))


def test_augmentation_opt_out_preserves_resize_and_eval_behavior():
    from dataclasses import replace
    from types import SimpleNamespace

    from openpi.models.model import Observation
    from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS

    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    assert OpenPi0Config.__dataclass_fields__["image_augmentation"].default is True
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(image_augmentation=False)
    obs = Observation(
        images={k: torch.full((2, 3, 224, 224), 0.2) for k in IMAGE_KEYS},
        image_masks={k: torch.ones(2, dtype=torch.bool) for k in IMAGE_KEYS},
        state=torch.zeros(2, 32),
        tokenized_prompt=torch.zeros(2, 8, dtype=torch.int64),
        tokenized_prompt_mask=torch.ones(2, 8, dtype=torch.bool),
    )
    rng = torch.get_rng_state().clone()
    images, *_ = model._preprocess_observation(obs, train=True)
    assert torch.equal(rng, torch.get_rng_state())
    for actual, expected in zip(images, obs.images.values()):
        torch.testing.assert_close(actual, expected)
    model.config.image_augmentation = True
    eval_images, *_ = model._preprocess_observation(obs, train=False)
    for actual, expected in zip(eval_images, obs.images.values()):
        torch.testing.assert_close(actual, expected)
    torch.manual_seed(42)
    augmented, *_ = model._preprocess_observation(obs, train=True)
    assert any(not torch.equal(a, b) for a, b in zip(augmented, images))
    model.config.image_augmentation = False
    obs = replace(obs, images={k: torch.zeros(2, 3, 240, 320) for k in IMAGE_KEYS})
    resized, *_ = model._preprocess_observation(obs, train=True)
    assert all(image.shape == (2, 3, 224, 224) for image in resized)
