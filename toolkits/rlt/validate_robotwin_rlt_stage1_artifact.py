#!/usr/bin/env python3
"""Validate a RoboTwin pi0 RLT Stage 1 endpoint on one fixed real-data batch.

This is an artifact acceptance probe, not another training experiment.  It:

1. constructs the original frozen pi0 + freshly initialized RLT module;
2. verifies that every non-RLT tensor in the endpoint equals the base model;
3. measures fresh and endpoint reconstruction loss on the same real prefixes;
4. compares true, batch-shuffled, and zero RL-token reconstruction; and
5. strictly reloads the full endpoint state dict in the current new process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--formal-resolved-config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--training-provenance", type=Path, required=True)
    parser.add_argument(
        "--manifest-id",
        default="robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="adjust the bottle")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _combined_batch_sha256(observation: Any, actions: torch.Tensor) -> str:
    digest = hashlib.sha256()
    named_tensors: list[tuple[str, torch.Tensor]] = [
        ("actions", actions),
        ("state", observation.state),
    ]
    for key, value in sorted(observation.images.items()):
        named_tensors.append((f"images/{key}", value))
    for key, value in sorted(observation.image_masks.items()):
        named_tensors.append((f"image_masks/{key}", value))
    if observation.tokenized_prompt is not None:
        named_tensors.append(("tokenized_prompt", observation.tokenized_prompt))
    if observation.tokenized_prompt_mask is not None:
        named_tensors.append(
            ("tokenized_prompt_mask", observation.tokenized_prompt_mask)
        )
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(_tensor_sha256(tensor).encode())
    return digest.hexdigest()


def _masked_mse(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    error = torch.square(
        reconstruction.to(dtype=torch.float32)
        - target.detach().to(dtype=torch.float32)
    )
    if mask is None:
        return error.mean()
    expanded = mask.to(device=error.device, dtype=error.dtype)[..., None]
    denominator = torch.clamp(expanded.sum() * target.shape[-1], min=1.0)
    return (error * expanded).sum() / denominator


def _model_cfg(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "model_path": str(args.base_model),
            "model_type": "openpi",
            "precision": None,
            "num_action_chunks": 10,
            "action_dim": 14,
            "add_value_head": False,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 4,
            "openpi_data": {
                "repo_id": "physical-intelligence/robotwin",
                "default_prompt": args.prompt,
                "norm_stats_path": str(args.norm_stats),
            },
            "openpi": {
                "config_name": "pi0_aloha_robotwin",
                "num_images_in_input": 3,
                "noise_level": 0.5,
                "action_horizon": 50,
                "action_chunk": 10,
                "action_env_dim": 14,
                "num_steps": 4,
                "state_indices": [],
                "train_expert_only": False,
                "add_value_head": False,
                "detach_critic_input": True,
                "noise_method": "flow_ode",
                "use_rlt": True,
                "rlt_train_vla": False,
                "rlt_alpha": 0.0,
                "rlt_input_dim": 2048,
                "rlt_embed_dim": 2048,
                "rlt_num_rl_tokens": 1,
                "rlt_prefix_seq_len": 768,
                "rlt_num_layers": 2,
                "rlt_num_heads": 8,
                "rlt_mlp_ratio": 4.0,
                "rlt_image_only": True,
                "rlt_use_mask": True,
                "rlt_action_adapter": "robotwin_aloha_canonical_v1",
            },
        }
    )


def _build_fixed_batch(args: argparse.Namespace):
    from openpi.training import data_loader as openpi_data_loader

    config = get_openpi_config(
        "pi0_aloha_robotwin",
        model_path=str(args.base_model),
        batch_size=args.batch_size,
        repo_id=str(args.dataset),
        data_kwargs={
            "repo_id": "physical-intelligence/robotwin",
            "default_prompt": args.prompt,
            "norm_stats_path": str(args.norm_stats),
        },
    )
    loader = openpi_data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
        num_batches=1,
    )
    observation, actions = next(iter(loader))
    if int(actions.shape[0]) != args.batch_size:
        raise AssertionError(
            f"fixed batch size mismatch: {actions.shape[0]} != {args.batch_size}"
        )
    return observation, actions


def _init_dist() -> bool:
    if dist.is_initialized():
        return False
    if "RANK" not in os.environ:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=1")
    dist.init_process_group("gloo")
    if dist.get_world_size() != 1:
        raise RuntimeError("artifact validation is intentionally single-rank")
    return True


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    initialized_dist = _init_dist()
    try:
        if args.batch_size < 2:
            raise ValueError("batch-size must be at least 2 for shuffled-z control")
        weights_path = (
            args.endpoint / "actor" / "model_state_dict" / "full_weights.pt"
        )
        required_paths = [
            args.endpoint,
            weights_path,
            args.base_model,
            args.norm_stats,
            args.dataset,
            args.dataset_manifest,
            args.formal_resolved_config,
            args.source_config,
            args.training_provenance,
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(path)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real-prefix probe")

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        model = get_model(_model_cfg(args)).eval()
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        unexpected_trainable = [
            name for name in trainable_names if not name.startswith("rlt_module.")
        ]
        if unexpected_trainable:
            raise AssertionError(
                f"non-RLT parameters are trainable: {unexpected_trainable[:10]}"
            )

        endpoint_state = torch.load(
            weights_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        fresh_state = model.state_dict()
        missing_keys = sorted(set(fresh_state) - set(endpoint_state))
        unexpected_keys = sorted(set(endpoint_state) - set(fresh_state))
        if missing_keys or unexpected_keys:
            raise AssertionError(
                "endpoint key-set mismatch: "
                f"missing={missing_keys[:10]}, unexpected={unexpected_keys[:10]}"
            )

        non_rlt_tensor_count = 0
        non_rlt_parameter_count = 0
        non_rlt_changed: list[str] = []
        rlt_tensor_count = 0
        rlt_changed: list[str] = []
        for name, fresh_tensor in fresh_state.items():
            endpoint_tensor = endpoint_state[name]
            if name.startswith("rlt_module."):
                rlt_tensor_count += 1
                if not torch.equal(fresh_tensor, endpoint_tensor):
                    rlt_changed.append(name)
            else:
                non_rlt_tensor_count += 1
                non_rlt_parameter_count += fresh_tensor.numel()
                if not torch.equal(fresh_tensor, endpoint_tensor):
                    non_rlt_changed.append(name)
        if non_rlt_changed:
            raise AssertionError(
                f"frozen pi0 tensor delta detected: {non_rlt_changed[:10]}"
            )

        observation, actions = _build_fixed_batch(args)
        batch_sha256 = _combined_batch_sha256(observation, actions)
        action_min = float(actions.min())
        action_max = float(actions.max())
        state_min = float(observation.state.min())
        state_max = float(observation.state.max())

        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        model = model.to(device).eval()

        with torch.inference_mode():
            fresh_prefix, fresh_mask = model._extract_rlt_prefix_embeddings(
                observation, train=False
            )
            rlt_parameter = next(model.rlt_module.parameters())
            fixed_prefix = fresh_prefix.to(
                device=rlt_parameter.device, dtype=rlt_parameter.dtype
            )
            fixed_mask = fresh_mask.to(device=rlt_parameter.device)
            fresh_loss, _ = model.rlt_module.loss(fixed_prefix, fixed_mask)

            incompatible = model.load_state_dict(endpoint_state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise AssertionError(str(incompatible))
            del endpoint_state

            endpoint_prefix, endpoint_mask = model._extract_rlt_prefix_embeddings(
                observation, train=False
            )
            endpoint_prefix = endpoint_prefix.to(
                device=fixed_prefix.device, dtype=fixed_prefix.dtype
            )
            endpoint_mask = endpoint_mask.to(device=fixed_mask.device)
            prefix_max_abs = float(
                (endpoint_prefix - fixed_prefix).abs().max().item()
            )
            if prefix_max_abs != 0.0 or not torch.equal(endpoint_mask, fixed_mask):
                raise AssertionError(
                    "frozen-prefix changed after endpoint load: "
                    f"max_abs={prefix_max_abs}"
                )

            endpoint_loss, endpoint_outputs = model.rlt_module.loss(
                fixed_prefix, fixed_mask
            )
            true_tokens = endpoint_outputs["z_rl"].reshape(
                args.batch_size,
                model.rlt_module.num_rl_tokens,
                model.rlt_module.embed_dim,
            )
            true_reconstruction = model.rlt_module.decode(
                true_tokens, fixed_prefix.shape[-2]
            )
            shuffled_reconstruction = model.rlt_module.decode(
                true_tokens.roll(1, dims=0), fixed_prefix.shape[-2]
            )
            zero_reconstruction = model.rlt_module.decode(
                torch.zeros_like(true_tokens), fixed_prefix.shape[-2]
            )
            true_loss = _masked_mse(
                true_reconstruction, fixed_prefix, fixed_mask
            )
            shuffled_loss = _masked_mse(
                shuffled_reconstruction, fixed_prefix, fixed_mask
            )
            zero_loss = _masked_mse(
                zero_reconstruction, fixed_prefix, fixed_mask
            )

        metrics = {
            "fresh_seed0_proxy_loss": float(fresh_loss.item()),
            "endpoint_loss": float(endpoint_loss.item()),
            "true_z_loss": float(true_loss.item()),
            "shuffled_z_loss": float(shuffled_loss.item()),
            "zero_z_loss": float(zero_loss.item()),
        }
        finite = all(np.isfinite(value) for value in metrics.values())
        gates = {
            "strict_full_state_reload": True,
            "frozen_pi0_exact_delta_zero": not non_rlt_changed,
            "fixed_prefix_reload_exact": prefix_max_abs == 0.0,
            "all_losses_finite": finite,
            "endpoint_better_than_fresh_proxy": (
                metrics["endpoint_loss"] < metrics["fresh_seed0_proxy_loss"]
            ),
            "true_z_better_than_shuffled_z": (
                metrics["true_z_loss"] < metrics["shuffled_z_loss"]
            ),
            "true_z_better_than_zero_z": (
                metrics["true_z_loss"] < metrics["zero_z_loss"]
            ),
        }
        accepted = all(gates.values())
        result = {
            "schema_version": 1,
            "accepted": accepted,
            "paths": {
                "endpoint": str(args.endpoint.resolve()),
                "full_weights": str(weights_path.resolve()),
                "base_model": str(args.base_model.resolve()),
                "norm_stats": str(args.norm_stats.resolve()),
                "dataset": str(args.dataset.resolve()),
            },
            "identity": {
                "full_weights_size": weights_path.stat().st_size,
                "full_weights_sha256": _sha256(weights_path),
                "norm_stats_sha256": _sha256(args.norm_stats),
                "fixed_batch_sha256": batch_sha256,
                "seed": args.seed,
                "batch_size": args.batch_size,
            },
            "reload_contract": {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "non_rlt_tensor_count": non_rlt_tensor_count,
                "non_rlt_parameter_count": non_rlt_parameter_count,
                "non_rlt_changed_tensor_count": len(non_rlt_changed),
                "rlt_tensor_count": rlt_tensor_count,
                "rlt_changed_tensor_count": len(rlt_changed),
                "rlt_unchanged_tensor_count": rlt_tensor_count - len(rlt_changed),
                "trainable_tensor_count": len(trainable_names),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "all_trainable_under_rlt_module": not unexpected_trainable,
            },
            "fixed_batch": {
                "actions_shape": list(actions.shape),
                "state_shape": list(observation.state.shape),
                "actions_min": action_min,
                "actions_max": action_max,
                "state_min": state_min,
                "state_max": state_max,
                "prefix_shape": list(fixed_prefix.shape),
                "prefix_mask_true": int(fixed_mask.sum().item()),
                "z_rl_shape": list(endpoint_outputs["z_rl"].shape),
                "z_rl_sha256": _tensor_sha256(endpoint_outputs["z_rl"]),
                "prefix_reload_max_abs": prefix_max_abs,
            },
            "metrics": {
                **metrics,
                "endpoint_over_fresh_proxy": (
                    metrics["endpoint_loss"]
                    / metrics["fresh_seed0_proxy_loss"]
                ),
                "true_over_shuffled": (
                    metrics["true_z_loss"] / metrics["shuffled_z_loss"]
                ),
                "true_over_zero": (
                    metrics["true_z_loss"] / metrics["zero_z_loss"]
                ),
            },
            "gates": gates,
            "resources": {
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "process_max_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "wall_seconds": time.monotonic() - started,
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dcp_dir = args.endpoint / "actor" / "dcp_checkpoint"
        dcp_files = []
        for path in sorted(dcp_dir.iterdir()):
            record = {
                "name": path.name,
                "size": path.stat().st_size,
            }
            if path.name == ".metadata":
                record["sha256"] = _sha256(path)
            dcp_files.append(record)
        artifact_manifest = {
            "schema_version": 1,
            "manifest_id": args.manifest_id,
            "accepted": accepted,
            "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "single-task low-budget RoboTwin RLT port; "
                "not equal-scale ManiSkill reproduction"
            ),
            "stage1": {
                "task": "adjust_bottle",
                "endpoint_step": 2000,
                "model_path": str(args.endpoint.resolve()),
                "full_weights": {
                    "path": str(weights_path.resolve()),
                    "size": weights_path.stat().st_size,
                    "sha256": result["identity"]["full_weights_sha256"],
                },
                "dcp_checkpoint": {
                    "path": str(dcp_dir.resolve()),
                    "files": dcp_files,
                },
            },
            "model_contract": {
                "base_model": str(args.base_model.resolve()),
                "norm_stats_path": str(args.norm_stats.resolve()),
                "norm_stats_sha256": result["identity"]["norm_stats_sha256"],
                "canonical_adapter_version": "robotwin_aloha_canonical_v1",
                "image_prefix_shape": [768, 2048],
                "z_rl_dim": 2048,
                "action_horizon": 50,
                "action_chunk": 10,
                "action_dim": 14,
                "frozen_pi0_exact_delta_zero": gates[
                    "frozen_pi0_exact_delta_zero"
                ],
            },
            "dataset": {
                "path": str(args.dataset.resolve()),
                "manifest_path": str(args.dataset_manifest.resolve()),
                "manifest_sha256": _sha256(args.dataset_manifest),
                "episodes": 50,
                "budget_label": (
                    "single-task low-budget RLT port; "
                    "not equal-scale ManiSkill reproduction"
                ),
            },
            "training_inputs": {
                "formal_resolved_config": {
                    "path": str(args.formal_resolved_config.resolve()),
                    "sha256": _sha256(args.formal_resolved_config),
                },
                "source_config": {
                    "path": str(args.source_config.resolve()),
                    "sha256": _sha256(args.source_config),
                },
                "run_provenance": {
                    "path": str(args.training_provenance.resolve()),
                    "sha256": _sha256(args.training_provenance),
                },
            },
            "acceptance": {
                "validation_path": str(args.output_json.resolve()),
                "validation_sha256": _sha256(args.output_json),
                "tool_path": str(Path(__file__).resolve()),
                "tool_sha256": _sha256(Path(__file__).resolve()),
                "fixed_batch_sha256": result["identity"]["fixed_batch_sha256"],
                "metrics": result["metrics"],
                "gates": gates,
            },
        }
        args.artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_manifest.write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not accepted:
            raise SystemExit(2)
    finally:
        if initialized_dist and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
