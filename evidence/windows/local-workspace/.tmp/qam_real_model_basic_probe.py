from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType


CONFIG_PATH = Path(
    "/root/autodl-tmp/RLinf_fastwam_rlinf/logs/"
    "20260728_dsrl_pi0_robotwin_n20_smoke_v1/"
    "FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml"
)
MODEL_PATH = Path(
    "/root/autodl-tmp/models/rlinf/"
    "RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
)


def make_fixed_env_obs() -> dict:
    batch = 1
    height = 480
    width = 640
    pixel_count = batch * height * width * 3
    base = (
        torch.arange(pixel_count, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(batch, height, width, 3)
    )
    left = torch.roll(base, shifts=37, dims=2)
    right = torch.flip(base, dims=(1,))
    wrists = torch.stack((left, right), dim=1)
    states = torch.linspace(-0.25, 0.25, 14, dtype=torch.float32).reshape(
        batch,
        14,
    )
    return {
        "main_images": base,
        "wrist_images": wrists,
        "extra_view_images": None,
        "states": states,
        "task_descriptions": ["adjust the bottle"],
    }


def configure_qam():
    assert CONFIG_PATH.is_file(), CONFIG_PATH
    assert MODEL_PATH.is_dir(), MODEL_PATH
    cfg = OmegaConf.load(CONFIG_PATH)
    with open_dict(cfg):
        cfg.actor.model.model_path = str(MODEL_PATH)
        cfg.actor.model.num_action_chunks = 20
        cfg.actor.model.action_dim = 14
        cfg.actor.model.add_value_head = False
        cfg.actor.model.openpi.action_chunk = 20
        cfg.actor.model.openpi.action_env_dim = 14
        cfg.actor.model.openpi.train_expert_only = True
        cfg.actor.model.openpi.add_value_head = False
        cfg.actor.model.openpi.use_dsrl = False
        cfg.actor.model.openpi.use_rlt = False
        cfg.actor.model.openpi.is_nft = False
        cfg.actor.model.openpi.use_qam = True
    return cfg


def dtype_numel(module: torch.nn.Module) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for parameter in module.parameters():
        counts[str(parameter.dtype)] += parameter.numel()
    return dict(sorted(counts.items()))


def compare_fine_to_behavior(model) -> tuple[bool, float]:
    pairs = [
        (
            model.qam_fine.expert_model,
            model.paligemma_with_expert.gemma_expert.model,
        )
    ]
    for name in model.qam_fine._projection_names:  # noqa: SLF001
        pairs.append((getattr(model.qam_fine, name), getattr(model, name)))

    exact = True
    max_abs = 0.0
    with torch.no_grad():
        for fine_module, behavior_module in pairs:
            fine_state = fine_module.state_dict()
            behavior_state = behavior_module.state_dict()
            assert fine_state.keys() == behavior_state.keys()
            for key in fine_state:
                fine_tensor = fine_state[key]
                behavior_tensor = behavior_state[key]
                assert fine_tensor.shape == behavior_tensor.shape
                delta = (
                    fine_tensor.float() - behavior_tensor.float()
                ).abs().max().item()
                max_abs = max(max_abs, delta)
                exact = exact and torch.equal(
                    fine_tensor.float(),
                    behavior_tensor.float(),
                )
    return exact, max_abs


def main() -> None:
    torch.manual_seed(20260731)
    np.random.seed(20260731)
    cfg = configure_qam()

    print("QAM_REAL_MODEL_LOAD_BEGIN=1", flush=True)
    load_start = time.perf_counter()
    model = get_model(cfg.actor.model)
    load_seconds = time.perf_counter() - load_start
    assert model is not None
    assert model.config.use_qam is True
    assert model.config.use_dsrl is False
    assert model.config.action_horizon == 50
    assert model.config.action_chunk == 20
    assert model.config.action_dim == 32
    assert model.config.action_env_dim == 14
    assert model._qam_fine_initialized is True  # noqa: SLF001

    fine_params = sum(parameter.numel() for parameter in model.qam_fine.parameters())
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names
    assert all(name.startswith("qam_fine.") for name in trainable_names)
    assert fine_params == sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert all("lm_head" not in name for name in trainable_names)
    exact_copy, copy_max_abs = compare_fine_to_behavior(model)
    assert exact_copy and copy_max_abs == 0.0

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    move_start = time.perf_counter()
    model = model.to(device)
    model.eval()
    torch.cuda.synchronize(device)
    move_seconds = time.perf_counter() - move_start

    conditioning_start = time.perf_counter()
    conditioning = model(
        forward_type=ForwardType.QAM_FLOW,
        operation="conditioning",
        env_obs=make_fixed_env_obs(),
    )
    torch.cuda.synchronize(device)
    conditioning_seconds = time.perf_counter() - conditioning_start
    assert conditioning["state"].shape == (1, 32)
    assert conditioning["prefix_output"].shape[:2] == (1, 816)
    assert conditioning["critic_feature"].shape[:2] == (1, 4)
    assert conditioning["block_lengths"] == (256, 256, 256, 48)
    assert torch.isfinite(conditioning["critic_feature"].float()).all()

    valid_counts = []
    start = 0
    for length in conditioning["block_lengths"]:
        stop = start + length
        valid_counts.append(
            int(conditioning["prefix_pad_masks"][:, start:stop].sum().item())
        )
        start = stop
    assert all(count > 0 for count in valid_counts)

    x_t = torch.linspace(
        -1.0,
        1.0,
        50 * 32,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 50, 32)
    time_qam = torch.tensor([0.375], device=device, dtype=torch.float32)
    velocity_start = time.perf_counter()
    with torch.no_grad():
        behavior_velocity = model(
            forward_type=ForwardType.QAM_FLOW,
            operation="velocity",
            state=conditioning["state"],
            x_t=x_t,
            time_qam=time_qam,
            prefix_pad_masks=conditioning["prefix_pad_masks"],
            past_key_values=conditioning["past_key_values"],
            route="behavior",
        )
        fine_velocity = model(
            forward_type=ForwardType.QAM_FLOW,
            operation="velocity",
            state=conditioning["state"],
            x_t=x_t,
            time_qam=time_qam,
            prefix_pad_masks=conditioning["prefix_pad_masks"],
            past_key_values=conditioning["past_key_values"],
            route="fine",
        )
    torch.cuda.synchronize(device)
    velocity_seconds = time.perf_counter() - velocity_start
    velocity_max_abs = (
        behavior_velocity.float() - fine_velocity.float()
    ).abs().max().item()
    velocity_mean_abs = (
        behavior_velocity.float() - fine_velocity.float()
    ).abs().mean().item()
    assert torch.isfinite(behavior_velocity).all()
    assert torch.isfinite(fine_velocity).all()
    assert velocity_max_abs <= 5e-4, velocity_max_abs

    result = {
        "config_source": str(CONFIG_PATH),
        "model_path": str(MODEL_PATH),
        "load_seconds": load_seconds,
        "move_seconds": move_seconds,
        "conditioning_seconds": conditioning_seconds,
        "two_velocity_seconds": velocity_seconds,
        "fine_parameters": fine_params,
        "fine_dtype_numel": dtype_numel(model.qam_fine),
        "trainable_parameter_names": len(trainable_names),
        "fine_behavior_parameter_copy_exact": exact_copy,
        "fine_behavior_parameter_copy_max_abs": copy_max_abs,
        "prefix_output_shape": list(conditioning["prefix_output"].shape),
        "critic_feature_shape": list(conditioning["critic_feature"].shape),
        "prefix_block_lengths": list(conditioning["block_lengths"]),
        "prefix_block_valid_counts": valid_counts,
        "behavior_fine_velocity_max_abs": velocity_max_abs,
        "behavior_fine_velocity_mean_abs": velocity_mean_abs,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    print("QAM_REAL_MODEL_BASIC_RESULT=" + json.dumps(result, sort_keys=True))
    print("QAM_REAL_MODEL_BASIC_OK=1", flush=True)


if __name__ == "__main__":
    main()
