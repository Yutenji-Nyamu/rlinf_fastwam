#!/usr/bin/env python3
"""Fail-closed Stage 1 artifact preflight for RoboTwin RLT Stage 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}.")


def resolve_full_weights(model_path: Path) -> Path:
    candidates = (
        model_path / "model_state_dict" / "full_weights.pt",
        model_path / "actor" / "model_state_dict" / "full_weights.pt",
    )
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one Stage 1 full_weights.pt under "
            f"{model_path}, found {matches}."
        )
    return matches[0]


def validate(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.manifest_path).resolve(strict=True)
    model_path = Path(args.stage1_model_path).resolve(strict=True)
    norm_stats_path = Path(args.norm_stats_path).resolve(strict=True)

    require_equal(
        "Stage 1 manifest SHA256",
        sha256_file(manifest_path),
        args.manifest_sha256,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_equal("Stage 1 accepted flag", manifest.get("accepted"), True)
    require_equal("Stage 1 manifest schema", manifest.get("schema_version"), 1)
    require_equal("Stage 1 manifest ID", manifest.get("manifest_id"), args.manifest_id)

    stage1 = manifest.get("stage1")
    if not isinstance(stage1, dict):
        raise ValueError("Stage 1 manifest is missing the stage1 mapping.")
    manifest_model_path = Path(stage1.get("model_path", "")).resolve(strict=True)
    require_equal("Stage 1 model path", manifest_model_path, model_path)

    model_contract = manifest.get("model_contract")
    if not isinstance(model_contract, dict):
        raise ValueError("Stage 1 manifest is missing the model_contract mapping.")
    expected_model_contract = {
        "norm_stats_sha256": args.norm_stats_sha256,
        "canonical_adapter_version": args.canonical_adapter_version,
        "action_horizon": args.action_horizon,
        "action_chunk": args.action_chunk,
        "action_dim": args.action_dim,
        "z_rl_dim": args.z_rl_dim,
        "image_prefix_shape": [args.prefix_seq_len, args.prefix_dim],
    }
    for key, expected_value in expected_model_contract.items():
        require_equal(
            f"Stage 1 model contract {key}",
            model_contract.get(key),
            expected_value,
        )

    require_equal(
        "norm_stats SHA256",
        sha256_file(norm_stats_path),
        args.norm_stats_sha256,
    )
    manifest_norm_path = Path(
        model_contract.get("norm_stats_path", "")
    ).resolve(strict=True)
    require_equal("norm_stats path", manifest_norm_path, norm_stats_path)

    full_weights_path = resolve_full_weights(model_path)
    full_weights = stage1.get("full_weights")
    if not isinstance(full_weights, dict):
        raise ValueError("Stage 1 manifest is missing full_weights metadata.")
    manifest_weights_path = Path(full_weights.get("path", "")).resolve(strict=True)
    require_equal("Stage 1 full-weights path", manifest_weights_path, full_weights_path)
    full_weights_size = full_weights_path.stat().st_size
    require_equal(
        "Stage 1 full-weights size",
        full_weights_size,
        int(full_weights.get("size", -1)),
    )
    full_weights_sha256 = sha256_file(full_weights_path)
    require_equal(
        "Stage 1 full-weights SHA256",
        full_weights_sha256,
        full_weights.get("sha256"),
    )

    return {
        "passed": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(manifest_path),
            "id": args.manifest_id,
            "sha256": args.manifest_sha256,
        },
        "stage1_model": {
            "path": str(model_path),
            "full_weights_path": str(full_weights_path),
            "full_weights_size": full_weights_size,
            "full_weights_sha256": full_weights_sha256,
        },
        "norm_stats": {
            "path": str(norm_stats_path),
            "sha256": args.norm_stats_sha256,
        },
        "model_contract": expected_model_contract,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--stage1-model-path", required=True)
    parser.add_argument("--norm-stats-path", required=True)
    parser.add_argument("--norm-stats-sha256", required=True)
    parser.add_argument("--canonical-adapter-version", required=True)
    parser.add_argument("--action-horizon", required=True, type=int)
    parser.add_argument("--action-chunk", required=True, type=int)
    parser.add_argument("--action-dim", required=True, type=int)
    parser.add_argument("--z-rl-dim", required=True, type=int)
    parser.add_argument("--prefix-seq-len", required=True, type=int)
    parser.add_argument("--prefix-dim", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight output: {output}")

    try:
        payload = validate(args)
    except Exception as exc:
        payload = {
            "passed": False,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        raise

    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
