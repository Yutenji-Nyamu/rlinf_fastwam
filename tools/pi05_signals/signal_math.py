"""Read-only, action-indexed NumPy signal calculations.

All public scores have shape [B, H]. Denoising arrays use [B, M, H, D]
and contain the M model-evaluation inputs/outputs, NOT the extra final
post-update latent. Invalid scores are NaN and have valid=False.

These are the study's explicit adaptations of the cited signals, not full
implementations of the source methods. In particular, tail-5 Fresco,
Norm's layer/time mean, and U-GROW 10-vs-5 are study conventions.
"""

from __future__ import annotations

import operator
from typing import NamedTuple

import numpy as np


class SignalScore(NamedTuple):
    score: np.ndarray
    valid: np.ndarray


def _array(value: np.ndarray, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-D array; got {result.shape}")
    return result


def _count(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, (bool, np.bool_)) or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _epsilon(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    return result


def _finish(score: np.ndarray, valid: np.ndarray) -> SignalScore:
    score = np.asarray(score, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(score)
    if score.ndim != 2 or valid.shape != score.shape:
        raise ValueError("internal score and valid arrays must both have shape [B, H]")
    return SignalScore(np.where(valid, score, np.nan), valid)


def _tail_vector_variance(values: np.ndarray, tail_n: int, name: str) -> SignalScore:
    values = _array(values, 4, name)
    tail_n = _count(tail_n, "tail_n")
    if values.shape[1] < tail_n:
        raise ValueError(f"{name} has fewer than tail_n={tail_n} model evaluations")
    tail = values[:, -tail_n:, :, :]
    valid = np.all(np.isfinite(tail), axis=(1, 3))
    with np.errstate(over="ignore", invalid="ignore"):
        centered = tail - np.mean(tail, axis=1, keepdims=True)
        # Mean squared vector deviation == sum of coordinate population variances.
        score = np.mean(np.sum(centered * centered, axis=-1), axis=1)
    return _finish(score, valid)


def dv_score(z_eval: np.ndarray, *, tail_n: int = 5) -> SignalScore:
    """Clean-estimate vector variance over the last tail_n forward evaluations."""
    return _tail_vector_variance(z_eval, tail_n, "z_eval[B,M,H,D]")


def fresco_score(x_eval: np.ndarray, *, tail_n: int = 5) -> SignalScore:
    """Raw-input vector variance on exactly the same evaluation window as DV.

    The tail length and coordinate-sum reduction are study conventions;
    Fresco Eq. 7 only specifies temporal token variance.
    """
    return _tail_vector_variance(x_eval, tail_n, "x_eval[B,M,H,D]")


def geo_full(velocities: np.ndarray, *, epsilon: float = 1e-8) -> SignalScore:
    """Per-action sum of adjacent velocity changes / mean velocity norm.

    Full-path GEO Eq. 11, retaining H instead of flattening the action chunk.
    Use actual model velocities, not stochastic sampler displacements.
    """
    velocity = _array(velocities, 4, "velocities[B,M,H,D]")
    epsilon = _epsilon(epsilon)
    if velocity.shape[1] < 2:
        raise ValueError("GEO requires at least two model evaluations")
    valid = np.all(np.isfinite(velocity), axis=(1, 3))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        changes = np.linalg.norm(np.diff(velocity, axis=1), axis=-1)
        speed = np.linalg.norm(velocity, axis=-1)
        score = np.sum(changes, axis=1) / (np.mean(speed, axis=1) + epsilon)
    return _finish(score, valid)


def geoaac_growth(
    velocities: np.ndarray,
    flow_times: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> SignalScore:
    """GeoAAC Eqs. 2/4/5/7/8: positive relative ACTION-prefix growth.

    flow_times[M] are times of the M velocity evaluations, in either strictly
    increasing or strictly decreasing order. Stage weights instead use the
    evaluation-index midpoint (j+0.5)/M. The prefix score need not be monotone.

    H index 0 has no previous nonempty prefix and is always NaN/False.
    A nonfinite action invalidates its own prefix and every subsequent prefix.
    This score is prefix-dependent, not an independent action uncertainty.
    """
    velocity = _array(velocities, 4, "velocities[B,M,H,D]")
    times = _array(flow_times, 1, "flow_times[M]")
    epsilon = _epsilon(epsilon)
    batch_size, steps, horizon, _ = velocity.shape
    if steps < 2 or times.shape != (steps,):
        raise ValueError("GeoAAC needs M >= 2 and exactly M evaluation times")
    delta = np.diff(times)
    if not np.all(np.isfinite(times)) or not (
        np.all(delta > 0) or np.all(delta < 0)
    ):
        raise ValueError("flow_times must be finite and strictly monotone")

    midpoint = (np.arange(steps - 1, dtype=np.float64) + 0.5) / steps
    stage_weight = midpoint * (1.0 - midpoint) ** 2
    stage_weight /= np.mean(stage_weight)
    finite_action = np.all(np.isfinite(velocity), axis=(1, 3))
    finite_prefix = np.logical_and.accumulate(finite_action, axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        acceleration = np.diff(velocity, axis=1) / (
            np.abs(delta)[None, :, None, None] + epsilon
        )
        prefix_acceleration = np.sqrt(
            np.cumsum(np.sum(acceleration * acceleration, axis=-1), axis=-1)
        )
        prefix_speed = np.sqrt(
            np.cumsum(np.sum(velocity * velocity, axis=-1), axis=-1)
        )
        prefix_score = steps * np.sum(
            stage_weight[None, :, None] * prefix_acceleration, axis=1
        ) / (np.sum(prefix_speed, axis=1) + epsilon)
        growth = (prefix_score[:, 1:] - prefix_score[:, :-1]) / (
            np.abs(prefix_score[:, :-1]) + epsilon
        )
    valid_prefix = finite_prefix & np.isfinite(prefix_score)
    score = np.full((batch_size, horizon), np.nan, dtype=np.float64)
    valid = np.zeros((batch_size, horizon), dtype=bool)
    score[:, 1:] = np.maximum(growth, 0.0)
    valid[:, 1:] = valid_prefix[:, 1:] & valid_prefix[:, :-1]
    return _finish(score, valid)


def shift_score(first_mean: np.ndarray, last_mean: np.ndarray) -> SignalScore:
    """log1p distance between first/last evaluation layer-mean representations.

    The observer supplies its chosen layer means, each [B,H,d]. This function
    performs no additional pooling or normalization over hidden coordinates.
    """
    first = _array(first_mean, 3, "first_mean[B,H,d]")
    last = _array(last_mean, 3, "last_mean[B,H,d]")
    if first.shape != last.shape:
        raise ValueError("SHIFT first_mean and last_mean must have identical shapes")
    valid = np.all(np.isfinite(first) & np.isfinite(last), axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        score = np.log1p(np.linalg.norm(last - first, axis=-1))
    return _finish(score, valid)


def norm_score(
    layer_norms: np.ndarray,
    *,
    tail_n: int = 5,
    deepest_layers: int = 3,
) -> SignalScore:
    """Mean of precomputed L2 norms: last 5 evaluations, deepest 3 layers.

    Input is [B,M,L,H] with layers ordered shallow to deep. Values must be
    nonnegative norms. This layer/time reduction is the study convention.
    """
    norms = _array(layer_norms, 4, "layer_norms[B,M,L,H]")
    tail_n = _count(tail_n, "tail_n")
    deepest_layers = _count(deepest_layers, "deepest_layers")
    if norms.shape[1] < tail_n or norms.shape[2] < deepest_layers:
        raise ValueError("Norm input has fewer evaluations/layers than requested")
    selected = norms[:, -tail_n:, -deepest_layers:, :]
    valid = np.all(np.isfinite(selected) & (selected >= 0), axis=(1, 2))
    with np.errstate(over="ignore", invalid="ignore"):
        score = np.mean(selected, axis=(1, 2))
    return _finish(score, valid)


def sr_window5(last_hidden: np.ndarray) -> SignalScore:
    """Uncentered stable rank in a fixed 5-action window at each position.

    Input [B,H,d] is the final evaluation's hidden representation. The centered
    window shifts inward at either boundary, keeping exactly 5 rows. No row
    normalization is performed. A single common scale before SVD is purely
    numerical and cancels exactly from stable rank. The zero matrix has
    undefined stable rank and returns NaN/False rather than an invented rank.
    """
    hidden = _array(last_hidden, 3, "last_hidden[B,H,d]")
    batch_size, horizon, _ = hidden.shape
    if horizon < 5:
        raise ValueError("SR-window5 requires at least five action positions")
    score = np.full((batch_size, horizon), np.nan, dtype=np.float64)
    valid = np.zeros((batch_size, horizon), dtype=bool)
    # Reuse boundary windows; never change their size or pad synthetic tokens.
    window_starts = np.clip(np.arange(horizon) - 2, 0, horizon - 5)
    for batch in range(batch_size):
        for start in np.unique(window_starts):
            matrix = hidden[batch, start : start + 5, :]
            if not np.all(np.isfinite(matrix)):
                continue
            scale = np.max(np.abs(matrix))
            if scale == 0:
                continue
            try:
                singular_values = np.linalg.svd(matrix / scale, compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            largest = singular_values[0]
            if not np.isfinite(largest) or largest <= 0:
                continue
            rank = np.sum((singular_values / largest) ** 2)
            positions = window_starts == start
            score[batch, positions] = rank
            valid[batch, positions] = np.isfinite(rank)
    return _finish(score, valid)


def ugrow_10_vs_5(
    actions_10: np.ndarray,
    actions_5: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> SignalScore:
    """Mean-coordinate p=2 disagreement, retaining every action position.

    Supply final chunks [B,H,D] from the SAME observation and initial noise,
    solved completely with 10 and 5 evaluations. The normal 10-step output is
    reused. The source paper uses 3/5; 10/5 is this study's cost convention.

    Each coordinate uses population std / (RMS + epsilon), NOT std/abs(mean).
    With two outputs a,b and epsilon=0 this equals |a-b|/sqrt(2*(a*a+b*b)).
    Two zero outputs receive 0 by the explicit additive-epsilon convention.
    """
    a = _array(actions_10, 3, "actions_10[B,H,D]")
    b = _array(actions_5, 3, "actions_5[B,H,D]")
    epsilon = _epsilon(epsilon)
    if a.shape != b.shape:
        raise ValueError("U-GROW final chunks must have identical shapes")
    valid = np.all(np.isfinite(a) & np.isfinite(b), axis=-1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # Halving before subtraction avoids overflow for large opposite signs;
        # hypot computes the RMS without squaring large raw coordinates.
        numerator = np.abs(a * 0.5 - b * 0.5)
        denominator = np.hypot(a / np.sqrt(2.0), b / np.sqrt(2.0))
        coordinate_score = numerator / (denominator + epsilon)
        score = np.mean(coordinate_score, axis=-1)
    return _finish(score, valid)
