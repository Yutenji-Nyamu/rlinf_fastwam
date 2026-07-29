#!/usr/bin/env python3
"""Probe the frozen RoboTwin pi0 prefix contract without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from openpi.models import model as openpi_model

import rlinf
from rlinf.models.embodiment.openpi import get_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle"
        ),
    )
    parser.add_argument(
        "--norm-stats-path",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/rlinf/"
            "RLinf-Pi0-RoboTwin-SFT-adjust_bottle/"
            "physical-intelligence/robotwin/norm_stats.json"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="adjust the bottle")
    parser.add_argument("--main-height", type=int, default=256)
    parser.add_argument("--main-width", type=int, default=320)
    parser.add_argument("--wrist-height", type=int, default=128)
    parser.add_argument("--wrist-width", type=int, default=160)
    parser.add_argument("--expected-hidden-width", type=int)
    parser.add_argument("--expected-image-tokens", type=int)
    parser.add_argument(
        "--with-rlt",
        action="store_true",
        help="Also instantiate the token module and verify its freeze/shape contract.",
    )
    parser.add_argument("--rlt-input-dim", type=int, default=2048)
    parser.add_argument("--rlt-embed-dim", type=int, default=2048)
    parser.add_argument("--rlt-prefix-seq-len", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"model path not found: {args.model_path}")
    if not args.norm_stats_path.is_file():
        raise FileNotFoundError(
            f"normalization stats not found: {args.norm_stats_path}"
        )
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError(
            "CUDA probe requested, but torch.cuda.is_available() is false"
        )

    cfg = OmegaConf.create(
        {
            "model_path": str(args.model_path),
            "model_type": "openpi",
            "precision": None,
            "num_action_chunks": 10,
            "action_dim": 14,
            "add_value_head": False,
            "is_lora": False,
            "lora_rank": 32,
            "openpi_data": {
                "repo_id": "physical-intelligence/robotwin",
                "default_prompt": args.prompt,
                "norm_stats_path": str(args.norm_stats_path),
            },
            "openpi": {
                "config_name": "pi0_aloha_robotwin",
                "num_images_in_input": 3,
                "action_horizon": 50,
                "action_chunk": 10,
                "action_env_dim": 14,
                "num_steps": 4,
                "train_expert_only": True,
                "detach_critic_input": True,
                "use_rlt": args.with_rlt,
                "rlt_train_vla": False,
                "rlt_alpha": 0.0,
                "rlt_input_dim": args.rlt_input_dim,
                "rlt_embed_dim": args.rlt_embed_dim,
                "rlt_num_rl_tokens": 1,
                "rlt_prefix_seq_len": args.rlt_prefix_seq_len,
                "rlt_num_layers": 2,
                "rlt_num_heads": 8,
                "rlt_mlp_ratio": 4.0,
                "rlt_image_only": True,
                "rlt_use_mask": True,
                "rlt_action_adapter": "robotwin_aloha_canonical_v1",
            },
        }
    )

    model = get_model(cfg).to(args.device).eval()
    env_obs = {
        "main_images": torch.zeros(
            (1, args.main_height, args.main_width, 3), dtype=torch.uint8
        ),
        "wrist_images": torch.zeros(
            (1, 2, args.wrist_height, args.wrist_width, 3), dtype=torch.uint8
        ),
        "extra_view_images": None,
        "states": torch.zeros((1, 14), dtype=torch.float32),
        "task_descriptions": [args.prompt],
    }

    with torch.inference_mode():
        policy_obs = model.obs_processor(env_obs)
        processed_obs = model.input_transform(policy_obs, transpose=False)
        processed_obs = model.precision_processor(processed_obs)
        observation = openpi_model.Observation.from_dict(processed_obs)
        prefix, prefix_mask, _, lang_tokens, state = model._build_rlt_prefix_cache(
            observation, train=False
        )
        image_prefix, image_mask = model._select_rlt_prefix_embeddings(
            prefix, prefix_mask, lang_tokens
        )

    result = {
        "rlinf_file": str(Path(rlinf.__file__).resolve()),
        "model_path": str(args.model_path.resolve()),
        "norm_stats_path": str(args.norm_stats_path.resolve()),
        "norm_stats_sha256": _sha256(args.norm_stats_path),
        "device": str(args.device),
        "full_prefix_shape": list(prefix.shape),
        "full_prefix_dtype": str(prefix.dtype),
        "full_prefix_mask_shape": list(prefix_mask.shape),
        "full_prefix_mask_true": int(prefix_mask.sum().item()),
        "language_token_shape": None
        if lang_tokens is None
        else list(lang_tokens.shape),
        "image_prefix_shape": list(image_prefix.shape),
        "image_prefix_mask_shape": list(image_mask.shape),
        "image_prefix_mask_true": int(image_mask.sum().item()),
        "processed_state_shape": list(state.shape),
        "processed_state_dtype": str(state.dtype),
    }
    if args.with_rlt:
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
                f"non-RLT parameters remain trainable: {unexpected_trainable[:20]}"
            )
        rlt_param = next(model.rlt_module.parameters())
        rlt_prefix = image_prefix.to(
            device=rlt_param.device,
            dtype=rlt_param.dtype,
        )
        rlt_mask = image_mask.to(device=rlt_param.device)
        with torch.inference_mode():
            reconstruction_loss, rlt_outputs = model.rlt_module.loss(
                rlt_prefix, rlt_mask
            )
            rlt_obs, decode_context = model.extract_rlt_obs(
                env_obs, return_decode_context=True
            )
            decoded_reference = model.decode_rlt_action(
                rlt_obs["ref_chunk"], decode_context
            )
            raw_template = decode_context["raw_action_template"]
            processed_state = decode_context["processed_state"]
            legacy_decoded = model.output_transform(
                {"actions": raw_template, "state": processed_state}
            )["actions"]
            legacy_decoded = legacy_decoded[
                :,
                : model.config.action_chunk,
                : model.config.action_env_dim,
            ]
            canonical_parity_max_abs = float(
                (decoded_reference - legacy_decoded).abs().max().item()
            )
            torch.testing.assert_close(
                decoded_reference,
                legacy_decoded,
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        result.update(
            {
                "rlt_z_shape": list(rlt_outputs["z_rl"].shape),
                "rlt_reconstruction_loss": float(reconstruction_loss.item()),
                "trainable_parameter_tensor_count": len(trainable_names),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "all_trainable_under_rlt_module": True,
                "canonical_reference_shape": list(rlt_obs["ref_chunk"].shape),
                "decoded_reference_shape": list(decoded_reference.shape),
                "canonical_legacy_decode_parity": True,
                "canonical_legacy_decode_max_abs": canonical_parity_max_abs,
            }
        )
    if (
        args.expected_hidden_width is not None
        and image_prefix.shape[-1] != args.expected_hidden_width
    ):
        raise AssertionError(
            "hidden-width mismatch: "
            f"{image_prefix.shape[-1]} != {args.expected_hidden_width}"
        )
    if (
        args.expected_image_tokens is not None
        and image_prefix.shape[-2] != args.expected_image_tokens
    ):
        raise AssertionError(
            "image-token mismatch: "
            f"{image_prefix.shape[-2]} != {args.expected_image_tokens}"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
