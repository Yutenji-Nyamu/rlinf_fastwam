from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    global_loader_batch = int(os.environ["LOADER_GLOBAL_BATCH"])
    expected_local_batch = global_loader_batch // world_size

    from openpi.training import data_loader as openpi_data_loader

    from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    dataset = os.environ["ROBOTWIN_RLT_CLEAN50_PATH"]
    model_path = os.environ["ROBOTWIN_PI0_BASE_PATH"]
    norm_stats = os.environ["ROBOTWIN_PI0_NORM_STATS_PATH"]
    config = get_openpi_config(
        "pi0_aloha_robotwin",
        model_path=model_path,
        batch_size=global_loader_batch,
        repo_id=dataset,
        data_kwargs={
            "repo_id": "physical-intelligence/robotwin",
            "default_prompt": "adjust the bottle",
            "norm_stats_path": norm_stats,
        },
    )
    loader = openpi_data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=False,
        num_batches=1,
    )
    observation, actions = next(iter(loader))

    torch_loader = loader._data_loader.torch_loader
    if torch_loader.batch_size != expected_local_batch:
        raise AssertionError(
            f"rank {rank}: local batch {torch_loader.batch_size}, expected {expected_local_batch}"
        )
    if int(actions.shape[0]) != expected_local_batch:
        raise AssertionError(
            f"rank {rank}: actions batch {actions.shape[0]}, expected {expected_local_batch}"
        )
    if not torch.isfinite(actions).all():
        raise ValueError(f"rank {rank}: non-finite actions")
    if not torch.isfinite(observation.state).all():
        raise ValueError(f"rank {rank}: non-finite state")

    image_shapes = {
        key: shape_of(value) for key, value in sorted(observation.images.items())
    }
    image_mask_shapes = {
        key: shape_of(value) for key, value in sorted(observation.image_masks.items())
    }
    result = {
        "rank": rank,
        "world_size": world_size,
        "configured_global_loader_batch": global_loader_batch,
        "actual_local_batch": torch_loader.batch_size,
        "sampler": type(torch_loader.sampler).__name__,
        "sampler_rank": getattr(torch_loader.sampler, "rank", None),
        "sampler_replicas": getattr(torch_loader.sampler, "num_replicas", None),
        "dataset_root": str(resolve_lerobot_dataset_root(dataset)),
        "dataset_len": len(torch_loader.dataset),
        "state_shape": shape_of(observation.state),
        "actions_shape": shape_of(actions),
        "image_shapes": image_shapes,
        "image_mask_shapes": image_mask_shapes,
        "tokenized_prompt_shape": shape_of(observation.tokenized_prompt),
        "tokenized_prompt_mask_shape": shape_of(observation.tokenized_prompt_mask),
        "state_dtype": str(observation.state.dtype),
        "actions_dtype": str(actions.dtype),
        "actions_min": float(torch.min(actions)),
        "actions_max": float(torch.max(actions)),
        "state_min": float(torch.min(observation.state)),
        "state_max": float(torch.max(observation.state)),
    }
    print(json.dumps(result, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    main()
