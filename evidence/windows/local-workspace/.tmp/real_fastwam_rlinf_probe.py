"""Real checkpoint probe for the Fast-WAM x RLinf integration."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import replace

import torch
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.models.embodiment.fastwam.robotwin_adapter import (
    adapt_robotwin_observation,
    denormalize_actions,
)


def make_observation() -> dict[str, object]:
    height, width = 72, 96
    grid = torch.arange(height * width * 3, dtype=torch.int64).reshape(
        height, width, 3
    )
    first = (grid % 256).to(torch.uint8)
    second = torch.flip(((grid * 3 + 17) % 256).to(torch.uint8), dims=(0,))
    main = torch.stack((first, second), dim=0)
    left = torch.stack((torch.flip(first, dims=(1,)), second), dim=0)
    right = torch.stack((second, torch.flip(first, dims=(0, 1))), dim=0)
    wrist = torch.stack((left, right), dim=1)
    states = torch.stack(
        (
            torch.linspace(-0.2, 0.2, 14),
            torch.linspace(0.15, -0.15, 14),
        )
    ).float()
    return {
        "main_images": main,
        "wrist_images": wrist,
        "states": states,
        "task_descriptions": [
            "adjust the bottle upright",
            "place the bottle at the target",
        ],
    }


def select_batch(observation: dict[str, object], indices: list[int]):
    return {
        "main_images": observation["main_images"][indices],
        "wrist_images": observation["wrist_images"][indices],
        "states": observation["states"][indices],
        "task_descriptions": [observation["task_descriptions"][i] for i in indices],
    }


torch.cuda.set_device(0)
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
resolved = OmegaConf.load(
    "/root/autodl-tmp/fastwam-rlinf-setup/resolved_smoke.yaml"
)
policy = get_model(resolved.actor.model)
assert policy is not None
inventory = {
    "trainable_names": [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ],
    "trainable_numel": sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    ),
    "total_numel": sum(parameter.numel() for parameter in policy.parameters()),
}
assert inventory["trainable_names"]
assert all(
    name.startswith("model.mot.mixtures.action.")
    for name in inventory["trainable_names"]
)

observation = make_observation()
single = select_batch(observation, [0])
image, proprio, prompts = adapt_robotwin_observation(
    single,
    policy.processor,
    device=policy.device,
    dtype=policy.model_dtype,
)
with torch.no_grad():
    context, context_mask = policy.model.encode_prompt(prompts)
    official_normalized = policy.model.infer_action(
        prompt=None,
        input_image=image,
        action_horizon=32,
        proprio=proprio,
        context=context,
        context_mask=context_mask,
        negative_prompt="",
        text_cfg_scale=1.0,
        num_inference_steps=10,
        sigma_shift=None,
        seed=0,
        rand_device="cpu",
        tiled=False,
    )["action"]
official_physical = denormalize_actions(
    official_normalized.unsqueeze(0), policy.processor
)[:, :24]
rlinf_single, _ = policy.predict_action_batch(single, mode="eval")
official_b1_max_abs = float((rlinf_single - official_physical).abs().max())
print(
    json.dumps(
        {"stage": "official_b1", "max_abs": official_b1_max_abs},
        sort_keys=True,
    ),
    flush=True,
)
assert torch.allclose(rlinf_single, official_physical, atol=2e-3, rtol=2e-3)

policy.config = replace(policy.config, model_forward_batch_size=2)
batch_chunk2, _ = policy.predict_action_batch(observation, mode="eval")
policy.config = replace(policy.config, model_forward_batch_size=1)
batch_chunk1, _ = policy.predict_action_batch(observation, mode="eval")
chunk_max_abs = float((batch_chunk2 - batch_chunk1).abs().max())
chunk_mean_abs = float((batch_chunk2 - batch_chunk1).abs().mean())
chunk_reference_abs = float(batch_chunk1.abs().max())
print(
    json.dumps(
        {
            "stage": "chunk_2_vs_1",
            "max_abs": chunk_max_abs,
            "mean_abs": chunk_mean_abs,
            "reference_abs_max": chunk_reference_abs,
        },
        sort_keys=True,
    ),
    flush=True,
)

policy.config = replace(policy.config, model_forward_batch_size=2)
permuted, _ = policy.predict_action_batch(
    select_batch(observation, [1, 0]), mode="eval"
)
permutation_max_abs = float((batch_chunk2 - permuted[[1, 0]]).abs().max())
permutation_mean_abs = float((batch_chunk2 - permuted[[1, 0]]).abs().mean())
print(
    json.dumps(
        {
            "stage": "permutation",
            "max_abs": permutation_max_abs,
            "mean_abs": permutation_mean_abs,
        },
        sort_keys=True,
    ),
    flush=True,
)

torch.manual_seed(20260717)
_, rollout_info = policy.predict_action_batch(observation, mode="train")
replay = rollout_info["forward_inputs"]
assert not torch.equal(replay["chains"][0, 0], replay["chains"][1, 0])
assert replay["denoise_inds"].unique().numel() == 1
policy.train(True)
recomputed = policy.default_forward(replay, compute_logprobs=True)
old_logprobs = rollout_info["prev_logprobs"]
new_logprobs = recomputed["logprobs"]
replay_max_abs = float((old_logprobs - new_logprobs.detach().cpu()).abs().max())
replay_mean_abs = float((old_logprobs - new_logprobs.detach().cpu()).abs().mean())
print(
    json.dumps(
        {
            "stage": "replay",
            "max_abs": replay_max_abs,
            "mean_abs": replay_mean_abs,
        },
        sort_keys=True,
    ),
    flush=True,
)
assert new_logprobs.requires_grad
assert torch.allclose(
    old_logprobs, new_logprobs.detach().cpu(), atol=3e-3, rtol=3e-3
)

metrics = {
    "official_b1_max_abs": official_b1_max_abs,
    "chunk_2_vs_1_max_abs": chunk_max_abs,
    "chunk_2_vs_1_mean_abs": chunk_mean_abs,
    "chunk_reference_abs_max": chunk_reference_abs,
    "permutation_max_abs": permutation_max_abs,
    "permutation_mean_abs": permutation_mean_abs,
    "replay_old_new_max_abs": replay_max_abs,
    "replay_old_new_mean_abs": replay_mean_abs,
    "shared_denoise_index": int(replay["denoise_inds"][0]),
    "train_initial_latents_equal": bool(
        torch.equal(replay["chains"][0, 0], replay["chains"][1, 0])
    ),
    "trainable_parameter_tensors": len(inventory["trainable_names"]),
    "trainable_numel": inventory["trainable_numel"],
    "total_numel": inventory["total_numel"],
    "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
    "elapsed_seconds": time.perf_counter() - started,
}
print(json.dumps(metrics, indent=2, sort_keys=True))
del recomputed, rollout_info, replay, policy
gc.collect()
torch.cuda.empty_cache()
