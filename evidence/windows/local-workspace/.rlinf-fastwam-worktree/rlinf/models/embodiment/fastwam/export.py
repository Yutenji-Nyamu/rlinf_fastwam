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

"""Export an RLinf Fast-WAM actor checkpoint for the official loader.

RLinf DCP checkpoints remain the training/resume source of truth.  This module
only converts the model state into the small top-level schema produced by
``FastWAM.save_checkpoint``.  It deliberately accepts one canonical module
tree and does not guess or repeatedly strip wrapper prefixes.
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

_GLOBAL_STEP_PATTERN = re.compile(r"global_step_(\d+)")
_MOT_PREFIX = "model.mot."
_PROPRIO_PREFIX = "model.proprio_encoder."
_FORBIDDEN_ALIAS_PREFIXES = (
    "model.video_expert.",
    "model.action_expert.",
    "model.dit.",
)
_ALLOWED_IGNORED_PREFIXES = (
    "model.vae.",
    "model.text_encoder.",
    # PPO critic state remains part of RLinf DCP/resume checkpoints, but the
    # official Fast-WAM deploy schema has no value-head field.
    "value_head.",
)


@dataclass(frozen=True)
class ResolvedActorCheckpoint:
    """A validated RLinf model-state source."""

    kind: Literal["dcp", "full_weights"]
    path: Path
    step: int | None


def _step_from_path(path: Path) -> int | None:
    for candidate in (path, *path.parents):
        match = _GLOBAL_STEP_PATTERN.fullmatch(candidate.name)
        if match is not None:
            return int(match.group(1))
    return None


def _validate_dcp_directory(path: Path) -> None:
    if not (path / ".metadata").is_file():
        raise FileNotFoundError(f"DCP metadata file not found under: {path}")
    if not any(path.glob("*.distcp")):
        raise FileNotFoundError(f"No DCP shard (*.distcp) found under: {path}")


def resolve_actor_checkpoint(path: str | Path) -> ResolvedActorCheckpoint:
    """Resolve one explicit RLinf actor checkpoint form.

    Accepted inputs are a ``global_step_N`` directory, a direct
    ``dcp_checkpoint`` directory, or a direct ``full_weights.pt`` file.
    Actor directories and arbitrary recursive searches are intentionally not
    supported.

    Args:
        path: Explicit checkpoint path.

    Returns:
        The validated checkpoint kind, concrete path, and path-derived step.

    Raises:
        FileNotFoundError: If the path or expected DCP files do not exist.
        ValueError: If the path is not one of the supported explicit forms.
    """

    source = Path(path).expanduser().resolve(strict=True)
    step = _step_from_path(source)

    if source.is_file():
        if source.name != "full_weights.pt":
            raise ValueError(
                "Only a direct full_weights.pt file is accepted as a file input; "
                f"got: {source}"
            )
        return ResolvedActorCheckpoint("full_weights", source, step)

    if source.name == "dcp_checkpoint":
        _validate_dcp_directory(source)
        return ResolvedActorCheckpoint("dcp", source, step)

    if _GLOBAL_STEP_PATTERN.fullmatch(source.name) is not None:
        dcp_path = source / "actor" / "dcp_checkpoint"
        if not dcp_path.is_dir():
            raise FileNotFoundError(
                "Expected actor/dcp_checkpoint under global-step directory: "
                f"{source}"
            )
        _validate_dcp_directory(dcp_path)
        return ResolvedActorCheckpoint("dcp", dcp_path, step)

    raise ValueError(
        "Checkpoint path must be a global_step_N directory, a direct "
        f"dcp_checkpoint directory, or full_weights.pt; got: {source}"
    )


def _load_dcp_model_state(path: Path) -> Mapping[str, Any]:
    # This is the exact no-distributed-process extraction strategy from
    # RLinf 6d0db56bf26f972cd27fa29535f5eb939e80e5bf:
    # rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py:54-63.
    # Imports are local because these private APIs are tied to the locked
    # Torch 2.7.1 environment and should not affect ordinary package imports.
    try:
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.format_utils import (
            _EmptyStateDictLoadPlanner,
        )
        from torch.distributed.checkpoint.state_dict_loader import (
            _load_state_dict,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "RLinf Fast-WAM DCP export requires the private DCP APIs present "
            "in the locked Torch 2.7.1 environment."
        ) from exc

    checkpoint: dict[str, Any] = {}
    _load_state_dict(
        checkpoint,
        storage_reader=FileSystemReader(str(path)),
        planner=_EmptyStateDictLoadPlanner(keys={"fsdp_checkpoint.model"}),
        no_dist=True,
    )
    try:
        model_state = checkpoint["fsdp_checkpoint"]["model"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "Could not find fsdp_checkpoint.model in DCP checkpoint at "
            f"{path}; top-level keys are {list(checkpoint)}"
        ) from exc
    if not isinstance(model_state, Mapping):
        raise TypeError(
            "DCP fsdp_checkpoint.model must be a mapping, got "
            f"{type(model_state).__name__}"
        )
    return model_state


def _load_torch_mapping(path: Path, description: str) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"{description} must be a mapping, got {type(payload).__name__}: " f"{path}"
        )
    return payload


def load_rlinf_model_state(
    checkpoint: ResolvedActorCheckpoint | str | Path,
) -> Mapping[str, Any]:
    """Load an RLinf model state without constructing an FSDP model.

    Args:
        checkpoint: A resolved checkpoint or one explicit supported path.

    Returns:
        The flat RLinf policy model state mapping on CPU.
    """

    resolved = (
        checkpoint
        if isinstance(checkpoint, ResolvedActorCheckpoint)
        else resolve_actor_checkpoint(checkpoint)
    )
    if resolved.kind == "dcp":
        return _load_dcp_model_state(resolved.path)
    return _load_torch_mapping(resolved.path, "RLinf full model state")


def _require_tensor_mapping(
    value: Any,
    *,
    name: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    for key, tensor in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} contains a non-string key: {key!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name}[{key!r}] must be a Tensor, got " f"{type(tensor).__name__}"
            )
    return value


def _compare_tensor_schema(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"{name} key mismatch: missing={missing}, extra={extra}")
    for key in sorted(expected_keys):
        actual_tensor = actual[key]
        expected_tensor = expected[key]
        if actual_tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"{name}[{key!r}] shape mismatch: "
                f"{tuple(actual_tensor.shape)} != {tuple(expected_tensor.shape)}"
            )
        if actual_tensor.dtype != expected_tensor.dtype:
            raise ValueError(
                f"{name}[{key!r}] dtype mismatch: "
                f"{actual_tensor.dtype} != {expected_tensor.dtype}"
            )


def validate_official_schema(
    payload: Mapping[str, Any],
    base_checkpoint: Mapping[str, Any],
) -> None:
    """Validate an export against an official Fast-WAM checkpoint schema.

    Values are intentionally not compared because the action expert may have
    been updated.  Logical keys, tensor shapes, and tensor dtypes must match
    exactly.

    Args:
        payload: Candidate official deploy payload.
        base_checkpoint: Original official checkpoint used as schema oracle.

    Raises:
        TypeError: If either checkpoint has invalid field types.
        ValueError: If keys, tensor shapes, or dtypes differ.
    """

    expected_mot = _require_tensor_mapping(
        base_checkpoint.get("mot"), name="base checkpoint mot"
    )
    has_proprio = "proprio_encoder" in base_checkpoint
    expected_top_level = {"mot", "step", "torch_dtype"}
    if has_proprio:
        expected_top_level.add("proprio_encoder")
    actual_top_level = set(payload)
    if actual_top_level != expected_top_level:
        raise ValueError(
            "official payload top-level key mismatch: "
            f"expected={sorted(expected_top_level)}, "
            f"actual={sorted(actual_top_level)}"
        )

    actual_mot = _require_tensor_mapping(payload["mot"], name="payload mot")
    _compare_tensor_schema(actual_mot, expected_mot, name="mot")

    if has_proprio:
        expected_proprio = _require_tensor_mapping(
            base_checkpoint["proprio_encoder"],
            name="base checkpoint proprio_encoder",
        )
        actual_proprio = _require_tensor_mapping(
            payload["proprio_encoder"], name="payload proprio_encoder"
        )
        _compare_tensor_schema(
            actual_proprio,
            expected_proprio,
            name="proprio_encoder",
        )

    base_dtype = base_checkpoint.get("torch_dtype")
    if not isinstance(base_dtype, str):
        raise TypeError(
            "base checkpoint torch_dtype must be a string, got "
            f"{type(base_dtype).__name__}"
        )
    if payload["torch_dtype"] != base_dtype:
        raise ValueError(
            "torch_dtype mismatch: " f"{payload['torch_dtype']!r} != {base_dtype!r}"
        )
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")


def extract_official_fastwam_payload(
    state_dict: Mapping[str, Any],
    base_checkpoint: Mapping[str, Any],
    step: int,
) -> dict[str, Any]:
    """Map one canonical RLinf policy tree to the official Fast-WAM schema.

    Only ``model.mot.*`` and optional ``model.proprio_encoder.*`` are exported.
    A top-level RLinf PPO ``value_head.*`` is intentionally retained in DCP but
    omitted from this deploy-only payload.
    Frozen VAE and text-encoder state is expected in a full RLinf checkpoint but
    intentionally omitted because the official loader obtains those components
    from the configured Wan model base.

    Args:
        state_dict: Flat RLinf policy state.
        base_checkpoint: Original official checkpoint used as schema oracle.
        step: Non-negative RLinf global step.

    Returns:
        An exact official Fast-WAM deploy payload.
    """

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")

    mot: dict[str, torch.Tensor] = {}
    proprio: dict[str, torch.Tensor] = {}
    forbidden_aliases: list[str] = []
    unexpected: list[str] = []

    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"RLinf state contains a non-string key: {key!r}")
        if key.startswith(_FORBIDDEN_ALIAS_PREFIXES):
            forbidden_aliases.append(key)
            continue
        if key.startswith(_MOT_PREFIX):
            logical_key = key[len(_MOT_PREFIX) :]
            target = mot
        elif key.startswith(_PROPRIO_PREFIX):
            logical_key = key[len(_PROPRIO_PREFIX) :]
            target = proprio
        elif key.startswith(_ALLOWED_IGNORED_PREFIXES):
            continue
        else:
            unexpected.append(key)
            continue

        if not logical_key:
            raise ValueError(f"Empty logical state key after canonical prefix: {key}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"RLinf state {key!r} must be a Tensor, got " f"{type(value).__name__}"
            )
        if logical_key in target:
            raise ValueError(f"Duplicate logical Fast-WAM state key: {logical_key!r}")
        target[logical_key] = value

    if forbidden_aliases:
        raise ValueError(
            "Forbidden Fast-WAM alias prefixes remain in RLinf state: "
            f"{sorted(forbidden_aliases)}"
        )
    if unexpected:
        raise ValueError(
            "Unexpected RLinf state-dict keys outside the frozen Fast-WAM "
            f"contract: {sorted(unexpected)}"
        )
    if not mot:
        raise ValueError(f"No canonical {_MOT_PREFIX} tensors were found")

    payload: dict[str, Any] = {
        "mot": mot,
        "step": step,
        "torch_dtype": base_checkpoint.get("torch_dtype"),
    }
    base_has_proprio = "proprio_encoder" in base_checkpoint
    if base_has_proprio:
        if not proprio:
            raise ValueError(
                "Base checkpoint has proprio_encoder but canonical RLinf state "
                "does not"
            )
        payload["proprio_encoder"] = proprio
    elif proprio:
        raise ValueError(
            "Canonical RLinf state has proprio_encoder but base checkpoint does not"
        )

    validate_official_schema(payload, base_checkpoint)
    return payload


def export_deploy_checkpoint(
    checkpoint_path: str | Path,
    base_checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    step: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Export an RLinf actor checkpoint for official Fast-WAM deployment.

    Args:
        checkpoint_path: Explicit RLinf global-step, DCP, or full-state path.
        base_checkpoint_path: Original official Fast-WAM checkpoint.
        output_path: Destination ``.pt`` path.
        step: Optional global-step override. Required if it cannot be parsed
            from ``checkpoint_path``.
        overwrite: Whether an existing destination may be atomically replaced.

    Returns:
        The resolved output path.
    """

    resolved = resolve_actor_checkpoint(checkpoint_path)
    effective_step = resolved.step if step is None else step
    if effective_step is None:
        raise ValueError(
            "Could not derive global step from checkpoint path; pass step explicitly"
        )

    base_path = Path(base_checkpoint_path).expanduser().resolve(strict=True)
    if not base_path.is_file():
        raise FileNotFoundError(f"Base checkpoint is not a file: {base_path}")
    output = Path(output_path).expanduser().resolve(strict=False)
    if output.suffix != ".pt":
        raise ValueError(f"Official Fast-WAM checkpoint must end in .pt: {output}")
    if output == base_path or output == resolved.path:
        raise ValueError("Output path must differ from all input checkpoint paths")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output checkpoint already exists: {output}")

    state_dict = load_rlinf_model_state(resolved)
    base_checkpoint = _load_torch_mapping(
        base_path, "Official Fast-WAM base checkpoint"
    )
    payload = extract_official_fastwam_payload(
        state_dict,
        base_checkpoint,
        effective_step,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output path already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an RLinf Fast-WAM DCP/full state for official deployment."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="global_step_N, dcp_checkpoint, or full_weights.pt path",
    )
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help="Original official Fast-WAM checkpoint used as schema oracle",
    )
    parser.add_argument("--output", required=True, help="Destination .pt path")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Global-step override when it is not encoded in the input path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the Fast-WAM deploy export CLI."""

    args = _build_parser().parse_args(argv)
    output = export_deploy_checkpoint(
        checkpoint_path=args.checkpoint,
        base_checkpoint_path=args.base_checkpoint,
        output_path=args.output,
        step=args.step,
        overwrite=args.overwrite,
    )
    print(f"Exported official Fast-WAM checkpoint: {output}")


if __name__ == "__main__":
    main()
