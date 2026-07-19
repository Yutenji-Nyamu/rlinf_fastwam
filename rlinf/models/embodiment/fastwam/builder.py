# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construct the official Fast-WAM model as an RLinf embodied policy."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger

logger = get_logger()


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _default_fastwam_config_dir() -> Path:
    explicit = os.environ.get("FASTWAM_CONFIG_DIR")
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        import fastwam

        path = Path(fastwam.__file__).resolve().parents[2] / "configs"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Fast-WAM config directory not found: {path}. Set FASTWAM_CONFIG_DIR."
        )
    return path


def _register_official_resolvers() -> None:
    for name, fn in (
        ("eval", eval),  # noqa: S307 - required by the pinned official config
        ("max", lambda value: max(value)),
        ("split", lambda value, index: value.split("/")[int(index)]),
    ):
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)


def compose_official_robotwin_cfg(cfg: DictConfig) -> DictConfig:
    """Compose the pinned official config without retaining Fast-WAM Hydra state."""

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    fastwam_cfg = _cfg_get(cfg, "fastwam", {}) or {}
    config_dir = (
        Path(_cfg_get(fastwam_cfg, "config_dir", None) or _default_fastwam_config_dir())
        .expanduser()
        .resolve()
    )
    config_name = str(_cfg_get(fastwam_cfg, "config_name", "sim_robotwin"))
    if config_name.endswith(".yaml"):
        config_name = config_name[:-5]
    task = str(
        _cfg_get(
            fastwam_cfg,
            "task",
            "robotwin_uncond_3cam_384_1e-4",
        )
    )
    overrides = [f"task={task}"]
    overrides.extend(list(_cfg_get(fastwam_cfg, "overrides", []) or []))

    _register_official_resolvers()
    global_hydra = GlobalHydra.instance()
    outer_hydra = global_hydra.hydra if global_hydra.is_initialized() else None
    if global_hydra.is_initialized():
        global_hydra.clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            official_cfg = compose(config_name=config_name, overrides=overrides)
    finally:
        if global_hydra.is_initialized():
            global_hydra.clear()
        if outer_hydra is not None:
            global_hydra.initialize(outer_hydra)
    return official_cfg


def canonicalize_fastwam_module_tree(model: torch.nn.Module) -> torch.nn.Module:
    """Keep ``mot`` as the only registered expert tree.

    Upstream registers the same modules as ``video_expert``, ``action_expert``,
    ``mot`` and ``dit``.  RLinf enumerates parameters with duplicate removal disabled
    for synchronization, so the compatibility aliases must be non-registering.
    """

    mot = model.mot
    video_expert = model.video_expert
    action_expert = model.action_expert
    dit = model.dit
    if video_expert is not mot.mixtures["video"]:
        raise ValueError("Fast-WAM video_expert is not mot.mixtures['video']")
    if action_expert is not mot.mixtures["action"]:
        raise ValueError("Fast-WAM action_expert is not mot.mixtures['action']")
    if dit is not mot:
        raise ValueError("Fast-WAM dit alias is not mot")

    for alias in ("video_expert", "action_expert", "dit"):
        registered = model._modules.get(alias)
        if registered is not None:
            del model._modules[alias]
    object.__setattr__(model, "video_expert", mot.mixtures["video"])
    object.__setattr__(model, "action_expert", mot.mixtures["action"])
    object.__setattr__(model, "dit", mot)
    assert_model_inventory(model)
    return model


def freeze_action_only(model: torch.nn.Module) -> None:
    """Freeze every component except the canonical action expert."""

    model.requires_grad_(False)
    model.mot.mixtures["action"].requires_grad_(True)
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or not all(
        name.startswith("mot.mixtures.action.") for name in trainable
    ):
        raise RuntimeError(
            "Fast-WAM action-only freeze produced an invalid trainable set: "
            f"{trainable[:20]}"
        )


def assert_model_inventory(model: torch.nn.Module) -> dict[str, list[str]]:
    """Assert unique parameter objects and canonical expert/state-dict prefixes."""

    names_by_id: dict[int, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        names_by_id[id(parameter)].append(name)
    duplicate_parameters = [names for names in names_by_id.values() if len(names) > 1]
    if duplicate_parameters:
        raise RuntimeError(
            "Fast-WAM still exposes duplicate registered parameter paths: "
            f"{duplicate_parameters[:10]}"
        )

    forbidden_prefixes = ("video_expert.", "action_expert.", "dit.")
    bad_state_keys = [
        key for key in model.state_dict() if key.startswith(forbidden_prefixes)
    ]
    if bad_state_keys:
        raise RuntimeError(
            "Fast-WAM state_dict still contains compatibility alias keys: "
            f"{bad_state_keys[:20]}"
        )
    return {
        "trainable": [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        "frozen": [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        ],
    }


def _validate_official_contract(
    model, processor, cfg: DictConfig, official_cfg
) -> None:
    action_dim = int(_cfg_get(cfg, "action_dim", 14))
    action_horizon = int(_cfg_get(cfg, "action_horizon", 32))
    action_chunks = int(_cfg_get(cfg, "num_action_chunks", 24))
    inference_steps = int(_cfg_get(cfg, "num_inference_steps", 10))
    sigma_shift = _cfg_get(cfg, "sigma_shift", None)

    official_horizon = int(official_cfg.data.train.num_frames) - 1
    if action_dim != 14 or int(model.action_expert.action_dim) != action_dim:
        raise ValueError("Fast-WAM RoboTwin requires the official 14D action expert")
    if action_horizon != 32 or official_horizon != action_horizon:
        raise ValueError(
            f"Fast-WAM RoboTwin action horizon mismatch: RLinf={action_horizon}, "
            f"official={official_horizon}"
        )
    if action_chunks != 24 or not (0 < action_chunks <= action_horizon):
        raise ValueError("First-version RoboTwin execution prefix must be N=24")
    if inference_steps != 10:
        raise ValueError("Released RoboTwin Fast-WAM checkpoint requires S=10")
    if sigma_shift is not None:
        raise ValueError(
            "Pinned RoboTwin run uses sigma_shift=null and scheduler shift=5.0"
        )
    if float(model.infer_action_scheduler.shift) != 5.0:
        raise ValueError("Pinned Fast-WAM action inference shift must be 5.0")
    if int(model.infer_action_scheduler.num_train_timesteps) != 1000:
        raise ValueError("Pinned Fast-WAM action scheduler must use 1000 train steps")
    if str(model.video_expert.video_attention_mask_mode) != "first_frame_causal":
        raise ValueError(
            "Fast-WAM action-only inference requires first_frame_causal video attention"
        )
    if model.proprio_dim is None or int(model.proprio_dim) != 14:
        raise ValueError("Fast-WAM RoboTwin proprio encoder must consume 14D qpos")
    if int(model.text_dim) != 4096:
        raise ValueError("Pinned Fast-WAM text context width must be 4096")
    if model.tokenizer is None or int(model.tokenizer.seq_len) != 128:
        raise ValueError("Pinned Fast-WAM tokenizer length must be 128")

    state_meta = list(processor.shape_meta["state"])
    action_meta = list(processor.shape_meta["action"])
    if len(state_meta) != 1 or state_meta[0]["key"] != "default":
        raise ValueError("Expected singleton state key 'default'")
    if len(action_meta) != 1 or action_meta[0]["key"] != "default":
        raise ValueError("Expected singleton action key 'default'")
    if int(state_meta[0]["shape"]) != 14 or int(action_meta[0]["shape"]) != 14:
        raise ValueError("Fast-WAM RoboTwin state/action stats must be 14D")
    if int(processor.num_output_cameras) != 3:
        raise ValueError("Fast-WAM RoboTwin requires three cameras")


def build_fastwam_policy(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
):
    """Instantiate official Fast-WAM, canonicalize it and wrap it for RLinf."""

    from fastwam.datasets.lerobot.utils.normalizer import (
        load_dataset_stats_from_json,
    )
    from hydra.utils import instantiate

    from .fastwam_policy import FastWAMPolicy, FastWAMPolicyConfig

    checkpoint_value = _cfg_get(cfg, "checkpoint_path", None) or _cfg_get(
        cfg, "model_path", None
    )
    stats_value = _cfg_get(cfg, "dataset_stats_path", None)
    if not checkpoint_value:
        raise ValueError("Fast-WAM checkpoint_path/model_path is required")
    if not stats_value:
        raise ValueError("Fast-WAM dataset_stats_path is required")
    checkpoint_path = Path(
        os.path.expandvars(os.path.expanduser(str(checkpoint_value)))
    ).resolve()
    stats_path = Path(
        os.path.expandvars(os.path.expanduser(str(stats_value)))
    ).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fast-WAM checkpoint not found: {checkpoint_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Fast-WAM dataset stats not found: {stats_path}")

    official_cfg = compose_official_robotwin_cfg(cfg)
    official_cfg.model.load_text_encoder = True
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = instantiate(
        official_cfg.model,
        model_dtype=torch_dtype,
        device=device,
    )
    model.load_checkpoint(str(checkpoint_path))

    processor = instantiate(official_cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    _validate_official_contract(model, processor, cfg, official_cfg)
    canonicalize_fastwam_module_tree(model)
    freeze_action_only(model)
    inventory = assert_model_inventory(model)

    add_value_head = bool(_cfg_get(cfg, "add_value_head", False))
    value_head_cfg = _cfg_get(cfg, "value_head", {}) or {}
    value_feature_source = str(
        _cfg_get(value_head_cfg, "feature_source", "video_cache_last_v_mean")
    )
    value_feature_dim = int(_cfg_get(value_head_cfg, "input_dim", 3072))
    value_hidden_sizes = tuple(
        int(size) for size in _cfg_get(value_head_cfg, "hidden_sizes", (1024, 512, 256))
    )
    value_activation = str(_cfg_get(value_head_cfg, "activation", "relu"))
    value_bias_last = bool(_cfg_get(value_head_cfg, "bias_last", True))
    detach_critic_input = bool(_cfg_get(value_head_cfg, "detach_input", True))

    value_head = None
    if add_value_head:
        expected_contract = {
            "feature_source": "video_cache_last_v_mean",
            "input_dim": 3072,
            "hidden_sizes": (1024, 512, 256),
            "activation": "relu",
            "bias_last": True,
            "detach_input": True,
        }
        actual_contract = {
            "feature_source": value_feature_source,
            "input_dim": value_feature_dim,
            "hidden_sizes": value_hidden_sizes,
            "activation": value_activation.lower(),
            "bias_last": value_bias_last,
            "detach_input": detach_critic_input,
        }
        if actual_contract != expected_contract:
            raise ValueError(
                "First-version Fast-WAM PPO value-head contract mismatch: "
                f"expected={expected_contract}, actual={actual_contract}"
            )
        # Match RLinf's mixed-precision embodied PPO heads: rollout and FSDP
        # actor must execute the critic in the same model dtype.  The wrapper
        # converts the scalar predictions back to FP32 for GAE/value loss.
        value_head = ValueHead(
            input_dim=value_feature_dim,
            hidden_sizes=value_hidden_sizes,
            output_dim=1,
            activation=value_activation,
            bias_last=value_bias_last,
        ).to(device=device, dtype=torch_dtype)

    policy_cfg = FastWAMPolicyConfig(
        action_dim=int(_cfg_get(cfg, "action_dim", 14)),
        action_horizon=int(_cfg_get(cfg, "action_horizon", 32)),
        num_action_chunks=int(_cfg_get(cfg, "num_action_chunks", 24)),
        num_inference_steps=int(_cfg_get(cfg, "num_inference_steps", 10)),
        sigma_shift=_cfg_get(cfg, "sigma_shift", None),
        text_cfg_scale=float(_cfg_get(cfg, "text_cfg_scale", 1.0)),
        negative_prompt=str(_cfg_get(cfg, "negative_prompt", "")),
        rand_device=str(_cfg_get(cfg, "rand_device", "cpu")),
        tiled=bool(_cfg_get(cfg, "tiled", False)),
        model_forward_batch_size=int(_cfg_get(cfg, "model_forward_batch_size", 2)),
        eval_seed=int(_cfg_get(cfg, "eval_seed", 0)),
        noise_level=float(_cfg_get(_cfg_get(cfg, "rl", {}), "noise_level", 0.1)),
        value_feature_source=value_feature_source,
        value_feature_dim=value_feature_dim,
        detach_critic_input=detach_critic_input,
    )
    policy = FastWAMPolicy(
        model=model,
        processor=processor,
        config=policy_cfg,
        value_head=value_head,
    )
    critic_parameters = (
        sum(parameter.numel() for parameter in policy.value_head.parameters())
        if hasattr(policy, "value_head")
        else 0
    )
    logger.info(
        "Built Fast-WAM RoboTwin policy | checkpoint=%s | stats=%s | "
        "trainable_parameters=%d | critic_parameters=%d | H/N/S=%d/%d/%d",
        checkpoint_path,
        stats_path,
        len(inventory["trainable"]),
        critic_parameters,
        policy_cfg.action_horizon,
        policy_cfg.num_action_chunks,
        policy_cfg.num_inference_steps,
    )
    return policy


__all__ = [
    "assert_model_inventory",
    "build_fastwam_policy",
    "canonicalize_fastwam_module_tree",
    "compose_official_robotwin_cfg",
    "freeze_action_only",
]
