# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Import a LeRobot pi0.5 checkpoint whose only model-layout wrapper is ``model.``."""

from __future__ import annotations

import argparse
import json
import pathlib

import safetensors.torch
from safetensors import safe_open

from rlinf.utils.ckpt_convertor.openpi import _core

ASSET_ID = pathlib.Path("physical-intelligence") / "robotwin"


def _validate_source_config(source_dir: pathlib.Path) -> dict:
    config = json.loads((source_dir / "config.json").read_text())
    expected = {
        "type": "pi05",
        "chunk_size": 50,
        "n_action_steps": 50,
        "num_inference_steps": 10,
        "max_action_dim": 32,
        "max_state_dim": 32,
        "use_relative_actions": False,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"unsupported LeRobot pi0.5 contract: {mismatches}")
    return config


def _target_contract() -> dict[str, tuple[tuple[int, ...], str]]:
    from openpi.models import pi0_config

    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    base = pi0_config.Pi0Config(
        pi05=True, action_horizon=50, discrete_state_input=True
    )
    model = OpenPi0ForRLActionPrediction(OpenPi0Config(**base.__dict__))
    return {
        key: (tuple(value.shape), str(value.dtype))
        for key, value in model.state_dict().items()
    }


def _source_contract(model_path: pathlib.Path) -> dict[str, tuple[tuple[int, ...], str]]:
    dtype_names = {
        "F32": "torch.float32",
        "BF16": "torch.bfloat16",
        "F16": "torch.float16",
        "I64": "torch.int64",
        "BOOL": "torch.bool",
    }
    contract = {}
    with safe_open(model_path, framework="pt", device="cpu") as source:
        for key in source.keys():
            if not key.startswith("model."):
                raise ValueError(f"source key lacks required single model. prefix: {key}")
            bare_key = key.removeprefix("model.")
            if bare_key in contract:
                raise ValueError(f"duplicate target key after prefix removal: {bare_key}")
            tensor = source.get_slice(key)
            source_dtype = str(tensor.get_dtype())
            contract[bare_key] = (
                tuple(tensor.get_shape()),
                dtype_names.get(source_dtype, source_dtype),
            )
    return contract


def _write_norm_stats(source_dir: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    processor = source_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    with safe_open(processor, framework="pt", device="cpu") as stats:
        norm_stats = {
            "state": {
                "mean": stats.get_tensor("observation.state.mean").tolist(),
                "std": stats.get_tensor("observation.state.std").tolist(),
            },
            "actions": {
                "mean": stats.get_tensor("action.mean").tolist(),
                "std": stats.get_tensor("action.std").tolist(),
            },
        }
    path = output_dir / ASSET_ID / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"norm_stats": norm_stats}, indent=2) + "\n")
    return path


def convert(input_model: str, output_model: str, source_revision: str) -> dict:
    source_dir = pathlib.Path(input_model).resolve()
    output_dir = pathlib.Path(output_model).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    source_config = _validate_source_config(source_dir)
    source_path = _core.resolve_model_safetensors(source_dir)
    source_contract = _source_contract(source_path)
    target_contract = _target_contract()
    missing = sorted(target_contract.keys() - source_contract.keys())
    unexpected = sorted(source_contract.keys() - target_contract.keys())
    shape_mismatch = sorted(
        key
        for key in source_contract.keys() & target_contract.keys()
        if source_contract[key][0] != target_contract[key][0]
    )
    dtype_mismatch = sorted(
        key
        for key in source_contract.keys() & target_contract.keys()
        if source_contract[key][1] != target_contract[key][1]
    )
    # The freshly constructed target is an architecture oracle, not a checkpoint
    # dtype oracle: Sidney intentionally stores transformer weights in BF16 while
    # projections remain FP32. Preserve that native mixed-precision state exactly.
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "strict target accounting failed: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={shape_mismatch[:5]}, dtype_mismatch={dtype_mismatch[:5]}"
        )

    source_state = safetensors.torch.load_file(str(source_path), device="cpu")
    converted = _core.strip_wrapper_prefix(source_state, cast_dtype=None)
    output_dir.mkdir(parents=True, exist_ok=True)
    _core.save_safetensors(converted, output_dir / "model.safetensors")
    norm_path = _write_norm_stats(source_dir, output_dir)

    manifest = {
        "format": "rlinf-native-openpi-pi05",
        "source": "SidneyXie/pi05_robotwin",
        "source_revision": source_revision,
        "source_config": {
            key: source_config[key]
            for key in (
                "type",
                "chunk_size",
                "n_action_steps",
                "num_inference_steps",
                "max_action_dim",
                "max_state_dim",
                "use_relative_actions",
            )
        },
        "conversion": "remove exactly one leading model. wrapper; preserve tensor values and dtypes",
        "source_keys": len(source_contract),
        "target_keys": len(target_contract),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatch,
        "dtype_mismatch_count": len(dtype_mismatch),
        "dtype_note": (
            "Source mixed BF16/FP32 is preserved; a fresh target model is FP32 "
            "before RLinf applies its maintained mixed-precision policy."
        ),
        "state_contract_digest": _core.state_dict_digest(converted),
        "norm_stats": str(norm_path.relative_to(output_dir)),
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return manifest


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--source-revision", required=True)


def run(args: argparse.Namespace) -> None:
    convert(args.input_model, args.output_model, args.source_revision)
