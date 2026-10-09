"""Historical EXPO signal contracts and reused BC-DVCA weight mapping.

Collection traces describe the selected parent VLA proposal. Replay retains only
executed physical positions. No current-policy probe runs inside SGD.
"""

from __future__ import annotations

import copy
from numbers import Integral

import torch

from rlinf.algorithms.norm_signal import NORM_SIGNAL_SPEC
from rlinf.algorithms.online_bc_dvac_controls import (
    apply_chunk_dropout,
    bc_controls_contract,
    effective_alphas,
)
from rlinf.algorithms.online_bc_dvac_two_level import compute_two_level_bc_weights
from rlinf.algorithms.ugrow_signal import UGROW_SIGNAL_SPEC

TRACE_FIELDS = ("raw", "query", "position", "parent", "edited", "base_version")


def validate_trace(trace: dict, count: int) -> dict:
    """Validate small per-physical-action trace once when writing/loading it."""
    if not isinstance(trace, dict) or set(trace) != set(TRACE_FIELDS):
        raise ValueError("Online signal trace fields differ")
    result = {}
    for key in TRACE_FIELDS:
        value = torch.as_tensor(trace[key]).detach().cpu()
        if value.shape != (count,) or not torch.isfinite(value).all():
            raise ValueError(
                "Signal trace must align with every executed physical action: " + key
            )
        if key == "raw":
            if not value.is_floating_point() or (value < 0).any():
                raise ValueError("Raw signal must be finite nonnegative float")
            value = value.float()
        elif key == "edited":
            if not ((value == 0) | (value == 1)).all():
                raise ValueError("Edited provenance must be binary")
            value = value.bool()
        else:
            if (
                value.is_floating_point()
                or value.dtype == torch.bool
                or (value < 0).any()
            ):
                raise ValueError("Signal provenance must contain nonnegative integers")
            value = value.long()
        result[key] = value.contiguous()
    if (result["position"] >= 10).any() or (result["parent"] >= 8).any():
        raise ValueError("Trace parent/position exceeds original N8/C10")
    if count and result["position"][0] != 0:
        raise ValueError("An episode trace must start at query position zero")
    if count > 1:
        same = result["query"][1:] == result["query"][:-1]
        if (
            (result["query"][1:] < result["query"][:-1]).any()
            or (result["position"][1:][same] != result["position"][:-1][same] + 1).any()
            or (result["position"][1:][~same] != 0).any()
        ):
            raise ValueError("Trace query/physical position order differs")
        for key in ("parent", "edited", "base_version"):
            if (result[key][1:][same] != result[key][:-1][same]).any():
                raise ValueError("Parent provenance changed inside one executed query")
    return result


class SignalWeighting:
    """One signal and independent FM/editor targets; disabled preserves Clean."""

    def __init__(self, cfg: dict | None = None):
        self.contract = None
        self.kind = self.target = None
        if not cfg:
            return
        allowed = {
            "kind",
            "target",
            "temperature_local",
            "temperature_chunk",
            "dropout",
            "dropout_probability",
            "anneal",
            "anneal_start_episode",
            "anneal_end_episode",
            "seed",
        }
        if set(cfg) - allowed:
            raise ValueError("Unknown EXPO signal configuration keys")
        self.kind, self.target = cfg.get("kind"), cfg.get("target")
        specs = {
            "ugrow_10_5": UGROW_SIGNAL_SPEC,
            "norm_residual_t5_l3": NORM_SIGNAL_SPEC,
        }
        if self.kind not in specs or self.target not in ("fm", "editor", "both"):
            raise ValueError("EXPO signal requires U/Norm and fm/editor/both")
        mapping = dict(
            normalization="two_level_batch",
            factor_mapping="exp_mean",
            alpha_local=1.0,
            alpha_chunk=1.0,
            chunk_dropout=dict(
                enabled=cfg.get("dropout", True),
                probability=cfg.get("dropout_probability", 0.2),
                seed=cfg.get("seed", 42),
            ),
            alpha_schedule=dict(enabled=cfg.get("anneal", True)),
        )
        for layer in ("local", "chunk"):
            mapping["alpha_schedule"][layer] = dict(
                enabled=True,
                start_step=cfg.get("anneal_start_episode", 1),
                end_step=cfg.get("anneal_end_episode", 200),
                end_alpha=0.0,
            )
        self.settings = dict(
            alpha_local=1.0,
            alpha_chunk=1.0,
            variance_eps=1e-12,
            range_eps=1e-6,
            factor_mapping="exp_mean",
            temperature_local=float(cfg.get("temperature_local", 2.5)),
            temperature_chunk=float(cfg.get("temperature_chunk", 2.5)),
        )
        compute_two_level_bc_weights(
            torch.ones(1, 1), torch.ones(1, 1), **self.settings
        )
        self.controls = bc_controls_contract(mapping)
        self.contract = dict(
            version=1,
            producer=copy.deepcopy(specs[self.kind]),
            target=self.target,
            source="collection_selected_parent_base; edited_j_has_parent_j",
            granularity=dict(
                fm="physical_action_inner_times_real_H50_window_outer",
                editor="real_C10_window_outer_full_alpha_logpi_minus_Q",
            ),
            demos="neutral_one_excluded_from_signal_normalization",
            settings=copy.deepcopy(self.settings),
            controls=copy.deepcopy(self.controls),
            schedule="completed_collection_episodes_including_warmup; R1=1 R200=0",
            dropout_counter="2*completed_update_calls + (0_fm_or_1_editor)",
        )

    def active(self, target: str) -> bool:
        return self.contract is not None and self.target in (target, "both")

    @torch.no_grad()
    def weights(
        self, batch: dict, target: str, *, completed_episodes: int, update_step: int
    ) -> tuple[torch.Tensor | None, dict]:
        """Map once on the complete sampled batch, before any device splitting."""
        if not self.active(target):
            return None, {}
        if (
            isinstance(completed_episodes, bool)
            or not isinstance(completed_episodes, Integral)
            or completed_episodes < 1
        ):
            raise ValueError("Weight schedule needs a completed collection episode")
        raw = batch["signal_raw"]
        valid = batch["signal_valid"]
        horizon = 50 if target == "fm" else 10
        if (
            raw.ndim != 2
            or raw.shape[1] != horizon
            or raw.shape != valid.shape
            or valid.dtype != torch.bool
        ):
            raise ValueError("Physical signal window must match target H50/C10")
        eligible = valid.all(-1)
        if (valid.any(-1) != eligible).any():
            raise ValueError(
                "Partially missing historical online trace is not supported"
            )
        settings = dict(self.settings)
        settings["alpha_local"], settings["alpha_chunk"] = effective_alphas(
            1.0 if target == "fm" else 0.0,
            1.0,
            self.controls,
            runner_step=completed_episodes - 1,
        )
        weights, metrics = compute_two_level_bc_weights(raw, valid, **settings)
        weights, dropped = apply_chunk_dropout(
            weights,
            eligible,
            self.controls,
            update_step=2 * update_step + int(target == "editor"),
        )
        metrics.update(
            alpha_local=settings["alpha_local"],
            alpha_chunk=settings["alpha_chunk"],
            completed_episodes=float(completed_episodes),
            eligible_windows=float(eligible.sum()),
            neutral_demo_windows=float((~eligible).sum()),
            dropout_windows=float(dropped.sum()),
            final_weight_mean=float(weights.mean()),
            final_weight_std=float(weights.std(unbiased=False)),
            final_weight_min=float(weights.min()),
            final_weight_max=float(weights.max()),
        )
        return (weights if target == "fm" else weights[:, 0]), metrics
