# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Cross-runtime action parity for a converted SidneyXie/pi05_robotwin model.

The native LeRobot and current RLinf environments cannot safely be imported in
one Python process.  This tool therefore uses one fixed NPZ input and two small
export commands, followed by a CPU-only comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _load_input(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    required = {"main_image", "wrist_images", "state", "noise", "prompt"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"input artifact is missing {sorted(missing)}")
    if result["main_image"].shape != (480, 640, 3):
        raise ValueError(
            f"expected main_image [480,640,3], got {result['main_image'].shape}"
        )
    if result["wrist_images"].shape != (2, 480, 640, 3):
        raise ValueError(
            f"expected wrist_images [2,480,640,3], got {result['wrist_images'].shape}"
        )
    if result["state"].shape != (14,):
        raise ValueError(f"expected state [14], got {result['state'].shape}")
    if result["noise"].shape != (1, 50, 32):
        raise ValueError(f"expected noise [1,50,32], got {result['noise'].shape}")
    result["prompt"] = str(result["prompt"].item())
    return result


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=value.dtype).contiguous()


def _to_bhwc(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"expected a batched image, got shape {tuple(image.shape)}")
    if image.shape[-1] == 3:
        return image
    if image.shape[1] == 3:
        return image.permute(0, 2, 3, 1)
    raise ValueError(f"cannot identify image channel axis in {tuple(image.shape)}")


def prepare(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    main_image = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    wrist_images = rng.integers(0, 256, size=(2, 480, 640, 3), dtype=np.uint8)
    state = np.linspace(-0.35, 0.35, 14, dtype=np.float32)
    noise = rng.standard_normal((1, 50, 32), dtype=np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        main_image=main_image,
        wrist_images=wrist_images,
        state=state,
        noise=noise,
        prompt=np.asarray(args.prompt),
    )
    print(f"wrote fixed parity input: {output}")


def export_native(args: argparse.Namespace) -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    raw = _load_input(args.input)
    device = torch.device(args.device)
    config = PreTrainedConfig.from_pretrained(args.model, local_files_only=True)
    if not isinstance(config, PI05Config):
        raise TypeError(f"expected PI05Config, got {type(config).__name__}")
    config.device = str(device)
    config.compile_model = False
    config.gradient_checkpointing = False
    policy = (
        PI05Policy.from_pretrained(
            args.model,
            config=config,
            local_files_only=True,
            strict=True,
        )
        .to(device)
        .eval()
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    batch = preprocessor(
        {
            OBS_STATE: torch.from_numpy(raw["state"]),
            "observation.images.cam_high": torch.from_numpy(raw["main_image"]).float()
            / 255.0,
            "observation.images.cam_left_wrist": torch.from_numpy(
                raw["wrist_images"][0]
            ).float()
            / 255.0,
            "observation.images.cam_right_wrist": torch.from_numpy(
                raw["wrist_images"][1]
            ).float()
            / 255.0,
            "task": raw["prompt"],
        }
    )
    images, image_masks = policy._preprocess_images(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    token_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
    noise = torch.from_numpy(raw["noise"]).to(device=device)
    with torch.no_grad():
        model_actions = policy.model.sample_actions(
            images,
            image_masks,
            tokens,
            token_mask,
            noise=noise,
            num_steps=10,
        )
        final_actions = postprocessor(model_actions[..., :14])

    normalized_state = batch[OBS_STATE]
    padded_state = torch.nn.functional.pad(
        normalized_state, (0, 32 - normalized_state.shape[-1])
    )
    payload = {
        "provenance": {
            "backend": "native-lerobot-v0.6",
            "model_path": str(Path(args.model).resolve()),
            "source_revision": args.source_revision,
        },
        "camera_order": CAMERA_ORDER,
        "images": _cpu(torch.stack([_to_bhwc(image) for image in images], dim=0)),
        "image_masks": _cpu(torch.stack(image_masks, dim=0)),
        "normalized_state14": _cpu(normalized_state),
        "padded_state32": _cpu(padded_state),
        "tokens": _cpu(tokens),
        "token_mask": _cpu(token_mask),
        "model_actions32": _cpu(model_actions),
        "final_actions14": _cpu(final_actions),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"wrote native LeRobot parity output: {args.output}")


def export_rlinf(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.openpi import get_model

    raw = _load_input(args.input)
    device = torch.device(args.device)
    config_path = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/model/pi0_5.yaml"
    )
    config = OmegaConf.load(config_path)
    config.model_path = args.model
    config.num_action_chunks = 50
    config.action_dim = 14
    config.num_steps = 10
    config.openpi.config_name = "pi05_sidney_robotwin"
    config.openpi.num_images_in_input = 3
    config.openpi.action_chunk = 50
    config.openpi.num_steps = 10
    config.openpi.noise_level = 0.3
    config.openpi.train_expert_only = True
    model = get_model(config).to(device).eval()

    env_obs = {
        "main_images": torch.from_numpy(raw["main_image"])[None].to(device),
        "wrist_images": torch.from_numpy(raw["wrist_images"])[None].to(device),
        "extra_view_images": None,
        "states": torch.from_numpy(raw["state"])[None].to(device),
        "task_descriptions": [raw["prompt"]],
    }
    policy_input = model.obs_processor(env_obs)
    processed = model.input_transform(policy_input, transpose=False)
    processed = model.precision_processor(processed)
    observation = openpi_model.Observation.from_dict(processed)
    noise = torch.from_numpy(raw["noise"]).to(device=device)
    with torch.no_grad():
        outputs = model.sample_actions(
            observation,
            noise=noise,
            mode="eval",
            compute_values=False,
        )
        final_actions = model.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"]

    image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    manifest = json.loads((Path(args.model) / "conversion_manifest.json").read_text())
    payload = {
        "provenance": {
            "backend": "current-rlinf-openpi",
            "model_path": str(Path(args.model).resolve()),
            "source_revision": manifest["source_revision"],
            "state_contract_digest": manifest["state_contract_digest"],
        },
        "camera_order": CAMERA_ORDER,
        "images": _cpu(
            torch.stack(
                [_to_bhwc(observation.images[key]) for key in image_keys], dim=0
            )
        ),
        "image_masks": _cpu(
            torch.stack([observation.image_masks[key] for key in image_keys], dim=0)
        ),
        "normalized_state14": _cpu(observation.state[..., :14]),
        "padded_state32": _cpu(observation.state),
        "tokens": _cpu(observation.tokenized_prompt),
        "token_mask": _cpu(observation.tokenized_prompt_mask),
        "model_actions32": _cpu(outputs["actions"]),
        "final_actions14": _cpu(final_actions),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"wrote current RLinf parity output: {args.output}")


def _compare_tensor(
    name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if left.shape != right.shape:
        raise AssertionError(
            f"{name}: shape {tuple(left.shape)} != {tuple(right.shape)}"
        )
    left_f = left.to(torch.float32)
    right_f = right.to(torch.float32)
    absolute = (left_f - right_f).abs()
    torch.testing.assert_close(left_f, right_f, rtol=rtol, atol=atol, msg=name)
    return {
        "shape": list(left.shape),
        "max_abs": float(absolute.max().item()) if absolute.numel() else 0.0,
        "mean_abs": float(absolute.mean().item()) if absolute.numel() else 0.0,
        "rtol": rtol,
        "atol": atol,
    }


def compare(args: argparse.Namespace) -> None:
    native = torch.load(args.native, map_location="cpu", weights_only=True)
    rlinf = torch.load(args.rlinf, map_location="cpu", weights_only=True)
    if (
        tuple(native["camera_order"]) != CAMERA_ORDER
        or tuple(rlinf["camera_order"]) != CAMERA_ORDER
    ):
        raise AssertionError("camera order does not match the three-camera contract")
    if (
        native["provenance"]["source_revision"]
        != rlinf["provenance"]["source_revision"]
    ):
        raise AssertionError(
            "native and converted outputs do not lock the same source revision"
        )
    if not rlinf["provenance"].get("state_contract_digest"):
        raise AssertionError("converted output has no state-contract digest")
    report = {
        "provenance": {
            "native_model": native["provenance"]["model_path"],
            "converted_model": rlinf["provenance"]["model_path"],
            "source_revision": native["provenance"]["source_revision"],
            "state_contract_digest": rlinf["provenance"]["state_contract_digest"],
        },
        "images": _compare_tensor(
            "images", native["images"], rlinf["images"], rtol=0.0, atol=1.0e-6
        ),
        "image_masks": _compare_tensor(
            "image_masks",
            native["image_masks"],
            rlinf["image_masks"],
            rtol=0.0,
            atol=0.0,
        ),
        "normalized_state14": _compare_tensor(
            "normalized_state14",
            native["normalized_state14"],
            rlinf["normalized_state14"],
            rtol=0.0,
            atol=1.0e-6,
        ),
        "padded_state32": _compare_tensor(
            "padded_state32",
            native["padded_state32"],
            rlinf["padded_state32"],
            rtol=0.0,
            atol=1.0e-6,
        ),
        "tokens": _compare_tensor(
            "tokens", native["tokens"], rlinf["tokens"], rtol=0.0, atol=0.0
        ),
        "token_mask": _compare_tensor(
            "token_mask", native["token_mask"], rlinf["token_mask"], rtol=0.0, atol=0.0
        ),
        "model_actions32": _compare_tensor(
            "model_actions32",
            native["model_actions32"],
            rlinf["model_actions32"],
            rtol=args.rtol,
            atol=args.atol,
        ),
        "final_actions14": _compare_tensor(
            "final_actions14",
            native["final_actions14"],
            rlinf["final_actions14"],
            rtol=args.rtol,
            atol=args.atol,
        ),
    }
    print(json.dumps(report, indent=2))


def _add_io(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--seed", type=int, default=1234)
    prepare_parser.add_argument("--prompt", default="move the stapler onto the pad")
    prepare_parser.set_defaults(func=prepare)

    native_parser = commands.add_parser("export-native")
    _add_io(native_parser)
    native_parser.add_argument("--source-revision", required=True)
    native_parser.set_defaults(func=export_native)

    rlinf_parser = commands.add_parser("export-rlinf")
    _add_io(rlinf_parser)
    rlinf_parser.set_defaults(func=export_rlinf)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--native", required=True)
    compare_parser.add_argument("--rlinf", required=True)
    compare_parser.add_argument("--rtol", type=float, default=1.0e-2)
    compare_parser.add_argument("--atol", type=float, default=5.0e-3)
    compare_parser.set_defaults(func=compare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
