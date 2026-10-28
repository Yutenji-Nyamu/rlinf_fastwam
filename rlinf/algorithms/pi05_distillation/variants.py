# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Opt-in physical-coordinate variants; historical objectives stay unchanged."""

import copy

V2_METHODS = {"cp_v2": "cp", "dmd_v2": "onedp", "mfd_v2": "mfd"}


def resolve_method(config):
    """Return the inference-compatible method and optional unified loss width."""
    name = config["method"]
    if name not in V2_METHODS:
        if "loss_coordinate_scope" in config:
            raise ValueError("Unified physical losses require an explicit v2 variant")
        return name, None
    if config.get("loss_coordinate_scope") != "physical":
        raise ValueError("v2 requires loss_coordinate_scope=physical")
    dim = config["model"]["openpi"]["action_env_dim"]
    if dim != 14 or config["model"]["action_dim"] != dim:
        raise ValueError(
            "These v2 variants require the established 14D Aloha interface"
        )
    for key in ("loss_action_dim", "onedp_loss_action_dim", "mfd_loss_action_dim"):
        if key in config and config[key] != dim:
            raise ValueError(f"Conflicting {key} for the physical v2 objective")
    return V2_METHODS[name], dim


def smoke_config(reference, variant):
    """Keep accepted model/data/batch/optimizer settings for a bounded 12->14 run."""
    if variant not in V2_METHODS or reference["method"] != V2_METHODS[variant]:
        raise ValueError("v2 reference must be the corresponding accepted method")
    result = copy.deepcopy(reference)
    result.update(
        method=variant,
        loss_coordinate_scope="physical",
        log_every=1,
        checkpoint_steps=100000,
        checkpoint_keep_last=1,
        checkpoint_keep_steps=[],
        smoke_inference_batch=32,
    )
    if variant == "dmd_v2":
        # The 192-sample smoke fragmented the default allocator at prefix build.
        # Keep batch/model mathematics; use the existing launcher allocation knob.
        result["cuda_allocator_conf"] = "expandable_segments:True"
    resolve_method(result)
    return result
