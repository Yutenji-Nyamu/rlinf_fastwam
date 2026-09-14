# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""High-V direct-PG selection with an independent, resumable full-update bypass.

No policy/rollout RNG is consumed. Selection is over native loss-valid positions;
the returned selected-mean weights belong on the existing logprob ST gradient
gate, never on forward logprob values or the loss denominator a second time.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DVACTop20Config:
    enabled: bool = False
    selection_domain: str = "batch"
    top_fraction: float = 0.2
    selected_l: int = 3
    full_update_probability: float = 0.0
    method_seed: int = 20260911

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "DVACTop20Config":
        values = {} if values is None else dict(values)
        unknown = set(values) - set(cls.__dataclass_fields__) - {"output_dir"}
        if unknown:
            raise ValueError(f"Unknown dvac_top20 configuration: {sorted(unknown)}")
        config = cls(**{key: value for key, value in values.items() if key != "output_dir"})
        if not isinstance(config.enabled, bool):
            raise ValueError("dvac_top20.enabled must be boolean")
        if config.selection_domain not in {"chunk", "batch"}:
            raise ValueError("dvac_top20.selection_domain must be chunk or batch")
        if isinstance(config.top_fraction, bool) or not isinstance(config.top_fraction, Real) or not math.isfinite(config.top_fraction) or not 0 < config.top_fraction <= 1:
            raise ValueError("dvac_top20.top_fraction must be in (0, 1]")
        if isinstance(config.full_update_probability, bool) or not isinstance(config.full_update_probability, Real) or not math.isfinite(config.full_update_probability) or not 0 <= config.full_update_probability <= 1:
            raise ValueError("dvac_top20.full_update_probability must be in [0, 1]")
        if isinstance(config.selected_l, bool) or not isinstance(config.selected_l, int) or config.selected_l < 2:
            raise ValueError("dvac_top20.selected_l must be an integer >= 2")
        if isinstance(config.method_seed, bool) or not isinstance(config.method_seed, int):
            raise ValueError("dvac_top20.method_seed must be an integer")
        return config

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm": "dvac_top20_logprob_st_v1",
            "normalization": "selected_mean",
            "tie_break": "blake2b_query_position_v1",
            "rng": "blake2b_counter_v1",
            "loss_mask": "native_clean_chunk_mask",
            **asdict(self),
        }


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _collective_device() -> torch.device:
    if _distributed() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("ascii"), digest_size=8).digest(), "big")


def expand_native_mask(variance: torch.Tensor, loss_mask: torch.Tensor | None) -> torch.Tensor:
    if variance.ndim != 2:
        raise ValueError(f"DVAC V must have shape [queries, horizon], got {variance.shape}")
    if loss_mask is None:
        return torch.ones_like(variance, dtype=torch.bool)
    if loss_mask.shape not in {variance.shape, (variance.shape[0], 1)}:
        raise ValueError(f"Native loss mask {loss_mask.shape} does not match V {variance.shape}")
    if loss_mask.dtype != torch.bool and not bool(((loss_mask == 0) | (loss_mask == 1)).all()):
        raise ValueError("Native loss mask must be boolean or binary")
    return loss_mask.bool().expand_as(variance)


def _validated_inputs(variance, valid, query_ids):
    values = variance.detach().to(device="cpu", dtype=torch.float32).contiguous()
    mask = expand_native_mask(values, valid.detach().cpu() if valid is not None else None)
    ids = query_ids.detach().cpu()
    if ids.ndim != 1 or ids.shape[0] != values.shape[0] or ids.dtype != torch.int64:
        raise ValueError("Stable query IDs must be int64 [queries]")
    if ids.unique().numel() != ids.numel():
        raise ValueError("Stable query IDs must be unique within the selection batch")
    active = values[mask]
    if not bool(torch.isfinite(active).all()) or not bool((active >= 0).all()):
        raise ValueError("Loss-valid DVAC V must be finite and nonnegative")
    return values, mask, ids


def compute_top20_weights(
    variance: torch.Tensor,
    valid: torch.Tensor | None,
    query_ids: torch.Tensor,
    *,
    top_fraction: float = 0.2,
    selection_domain: str = "batch",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact-k selection; stable tie order is independent of tensor shuffling.

    A deterministic query/position hash breaks exact V ties without consuming
    RNG or always choosing the temporal prefix. Invalid positions remain zero.
    The function works on CPU detached signal data, not model activations.
    """
    DVACTop20Config.from_dict({"top_fraction": top_fraction, "selection_domain": selection_domain})
    values, mask, ids = _validated_inputs(variance, valid, query_ids)
    horizon = values.shape[1]
    weights = torch.zeros_like(values)
    flat_values, flat_mask, flat_weights = values.flatten(), mask.flatten(), weights.flatten()
    domains = [torch.nonzero(flat_mask, as_tuple=False).flatten()]
    if selection_domain == "chunk":
        domains = [torch.nonzero(mask[row], as_tuple=False).flatten() + row * horizon for row in range(len(values))]
    boundary_ties = 0
    constant_domains = 0
    domain_count = 0
    for eligible in domains:
        count = eligible.numel()
        if not count:
            continue
        domain_count += 1
        k = min(count, max(1, math.ceil(top_fraction * count)))
        # Sort only the compact detached signal; final index key guards hash collisions.
        ranked = sorted(
            eligible.tolist(),
            key=lambda index: (
                -float(flat_values[index]),
                _hash64(f"dvac-tie-v1:{int(ids[index // horizon])}:{index % horizon}"),
                int(ids[index // horizon]),
                index % horizon,
            ),
        )
        chosen = torch.tensor(ranked[:k], dtype=torch.int64)
        flat_weights[chosen] = count / k
        constant_domains += int(float(flat_values[ranked[0]]) == float(flat_values[ranked[-1]]))
        if k < count:
            boundary_ties += int(float(flat_values[ranked[k - 1]]) == float(flat_values[ranked[k]]))
    selected = weights > 0
    n_valid, n_selected = int(mask.sum()), int(selected.sum())
    valid_rows = mask.any(dim=1)
    selected_rows = selected.sum(dim=1)
    selected_values = values[selected]
    metrics = {
        "valid_positions": float(n_valid),
        "selected_positions": float(n_selected),
        "selected_fraction": n_selected / n_valid if n_valid else 0.0,
        "top_weight_mean": float(weights[mask].mean()) if n_valid else 0.0,
        "top_weight_sq_mean": float(weights[mask].square().mean()) if n_valid else 0.0,
        "empty_chunk_fraction": float((selected_rows[valid_rows] == 0).float().mean()) if bool(valid_rows.any()) else 0.0,
        "constant_domain_fraction": constant_domains / domain_count if domain_count else 0.0,
        "boundary_tie_fraction": boundary_ties / domain_count if domain_count else 0.0,
        "valid_v_mean": float(values[mask].mean()) if n_valid else 0.0,
        "selected_v_mean": float(selected_values.mean()) if n_selected else 0.0,
        "selected_position_mean": float(torch.nonzero(selected)[:, 1].float().mean()) if n_selected else 0.0,
    }
    return weights, metrics


def distributed_top20_weights(variance, valid, query_ids, config):
    """Use all actor ranks' complete Adam input, before microbatch splitting.

    all_gather_object carries only CPU V/mask/IDs (about 250 KiB at B1024/H50).
    Local validation errors are exchanged before raising, avoiding peers waiting
    in a subsequent collective when one shard has invalid signals.
    """
    try:
        values, mask, ids = _validated_inputs(variance, valid, query_ids)
        payload = {"values": values, "mask": mask, "ids": ids, "error": None}
    except (ValueError, TypeError, RuntimeError) as error:
        payload = {"error": f"{type(error).__name__}: {error}"}
    gathered = [payload]
    rank = 0
    if _distributed():
        rank = dist.get_rank()
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)
    failures = [f"rank {index}: {part['error']}" for index, part in enumerate(gathered) if part["error"]]
    if failures:
        raise ValueError("DVAC top20 input validation failed: " + "; ".join(failures))
    sizes = [part["values"].shape[0] for part in gathered]
    all_values = torch.cat([part["values"] for part in gathered], dim=0)
    all_mask = torch.cat([part["mask"] for part in gathered], dim=0)
    all_ids = torch.cat([part["ids"] for part in gathered], dim=0)
    # Gathering both domains also makes diagnostics globally count-correct when
    # native masks remove different amounts of data on different actor ranks.
    weights, metrics = compute_top20_weights(
        all_values, all_mask, all_ids,
        top_fraction=config.top_fraction,
        selection_domain=config.selection_domain,
    )
    offset = sum(sizes[:rank])
    return weights[offset:offset + sizes[rank]].contiguous(), metrics


class DVACTop20State:
    """One independent counter-based Bernoulli decision per actual Adam slot.

    Only rank zero generates the decision; broadcasting the decision and next
    counter gives all microbatches/ranks the same gate. Seed/counter/algorithm
    identity completely specify continuation and are saved in a new sidecar.
    """

    def __init__(self, config: DVACTop20Config):
        self.config = config
        self.update_index = 0
        self.full_updates = 0

    def next_full_update(self) -> bool:
        rank = dist.get_rank() if _distributed() else 0
        value = False
        if rank == 0:
            random_integer = _hash64(f"dvac-full-v1:{self.config.method_seed}:{self.update_index}")
            threshold = int(self.config.full_update_probability * (1 << 64))
            value = random_integer < threshold
        decision = torch.tensor(
            [int(value), self.update_index + 1], dtype=torch.int64, device=_collective_device()
        )
        if _distributed():
            dist.broadcast(decision, src=0)
        result, next_index = decision.cpu().tolist()
        if next_index != self.update_index + 1:
            raise RuntimeError("DVAC method RNG update counters disagree across ranks")
        self.update_index = int(next_index)
        self.full_updates += int(result)
        return bool(result)

    def state_dict(self) -> dict[str, Any]:
        return {
            "identity": self.config.identity(),
            "update_index": self.update_index,
            "full_updates": self.full_updates,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("identity") != self.config.identity():
            raise ValueError("DVAC top20 checkpoint method identity mismatch")
        update_index, full_updates = payload.get("update_index"), payload.get("full_updates")
        if type(update_index) is not int or type(full_updates) is not int or not 0 <= full_updates <= update_index:
            raise ValueError("Invalid DVAC top20 method RNG counters")
        self.update_index, self.full_updates = update_index, full_updates
