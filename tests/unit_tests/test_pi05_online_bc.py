# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from pathlib import Path

import openpi.transforms as transforms
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from openpi.training.config import DataConfig

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.dataconfig import (
    robotwin_aloha_dataconfig as aloha_data,
)


def test_pi05_bc_inherits_learning_and_capacity_budget(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    for key, value in {
        "REPO_PATH": str(root),
        "EMBODIED_PATH": str(root / "examples/embodiment"),
        "ASSETS_PATH": str(tmp_path),
        "PI0_MODEL_PATH": str(tmp_path / "pi0"),
        "PI05_MODEL_PATH": str(tmp_path / "pi05"),
        "ONLINE_BC_RUN_DIR": str(tmp_path / "run"),
    }.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        bc = compose(config_name="robotwin_adjust_bottle_online_bc_openpi")
        pi05 = compose(
            config_name="robotwin_adjust_bottle_online_bc_openpi",
            overrides=["+online_bc_model=pi05_sidney"],
        )
    bc = OmegaConf.create(OmegaConf.to_container(bc, resolve=True))
    pi05 = OmegaConf.create(OmegaConf.to_container(pi05, resolve=True))
    for key in (
        "algorithm",
        "actor.optim",
        "actor.fsdp_config",
        "cluster",
        "actor.micro_batch_size",
        "actor.global_batch_size",
    ):
        assert OmegaConf.to_container(
            OmegaConf.create({"v": OmegaConf.select(bc, key)}), resolve=True
        ) == OmegaConf.to_container(
            OmegaConf.create({"v": OmegaConf.select(pi05, key)}), resolve=True
        )
    assert pi05.actor.model.openpi.config_name == "pi05_sidney_robotwin"
    assert pi05.actor.model.num_steps == pi05.actor.model.openpi.num_steps == 10
    assert pi05.actor.model.openpi.train_expert_only
    assert not pi05.actor.model.openpi.image_augmentation
    assert (
        pi05.env.train.task_config.task_name
        == pi05.env.eval.task_config.task_name
        == "move_pillbottle_pad"
    )
    assert (pi05.env.train.total_num_envs, pi05.env.train.rollout_epoch) == (32, 1)
    assert (pi05.env.eval.total_num_envs, pi05.env.eval.rollout_epoch) == (8, 4)
    assert pi05.env.eval.fixed_reset_batch_count == 4
    assert pi05.env.train.task_config.step_lim == 200
    assert (
        pi05.actor.model.num_action_chunks == 50 and pi05.actor.model.action_dim == 14
    )


def test_sidney_model_contract_and_opt_in_normalization(monkeypatch, tmp_path):
    config = get_openpi_config("pi05_sidney_robotwin", model_path=str(tmp_path))
    assert config.model.pi05 and config.model.discrete_state_input
    assert config.model.max_token_len == 200 and config.model.action_horizon == 50
    assert not config.data.adapt_to_pi and not config.data.extra_delta_transform
    assert config.data.assets.asset_id == "physical-intelligence/robotwin"
    assert config.data.assets.assets_dir == str(tmp_path)
    # Avoid tokenizer/assets I/O in this structural unit test; deployment validates
    # the real transforms and local norm statistics separately.
    monkeypatch.setattr(
        aloha_data,
        "ModelTransformFactory",
        lambda **kwargs: lambda model: transforms.Group(),
    )
    monkeypatch.setattr(
        aloha_data.LeRobotAlohaDataConfig,
        "create_base_config",
        lambda self, assets, model: DataConfig(use_quantile_norm=model.pi05),
    )
    assert not config.data.create(tmp_path, config.model).use_quantile_norm
    default = replace(config.data, use_quantile_norm=None)
    assert default.create(tmp_path, config.model).use_quantile_norm
    assert not default.create(
        tmp_path, replace(config.model, pi05=False)
    ).use_quantile_norm
    result = config.data.create(tmp_path, config.model)
    assert not any(
        isinstance(x, transforms.DeltaActions) for x in result.data_transforms.inputs
    )
    assert not any(
        isinstance(x, transforms.AbsoluteActions)
        for x in result.data_transforms.outputs
    )
