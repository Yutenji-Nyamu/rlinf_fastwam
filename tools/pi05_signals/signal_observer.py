"""Read-only residual-stream telemetry for the pi0.5 main inference chain.

The cached-prefix evaluation path must call each expert decoder block exactly
once per denoising forward. Hooks observe ``output[0]`` (the complete block
residual, before the model's final norm), and never replace an output.

Usage::

    observer = SignalObserver(model)  # model.config.action_horizon; runtime L
    try:
        for idx in range(10):
            with observer.step(idx, 10):
                result = model.sample_mean_var_val(...)  # one main forward
        telemetry = observer.flush()  # owned NumPy arrays, batch axis first
        # M3/M5 side chains run outside observer.step: their hooks are inactive.
    finally:
        observer.close()

This is a single-query, single-thread observer. It does not run a model, alter
RNG state, set eval mode, or disable gradients for the caller's forward. Only
the detached telemetry operations run under no_grad. Training/checkpoint
recomputation and compiled paths that bypass decoder-module hooks are not
supported; missing or duplicate calls fail validation instead of being mixed
silently. reset() discards a query but keeps hooks; close() removes hooks and
permanently closes the observer.
"""

from __future__ import annotations

from contextlib import contextmanager
import operator
from typing import Any, Iterator

import numpy as np
import torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


class SignalObserver:
    """Collect SHIFT inputs, per-layer norms, and last-layer hidden snapshots.

    Outputs from flush()/finish(), all NumPy arrays:

    * last_layer_hidden: float32 [B, M, H, d], every main-chain forward.
    * layer_norms: float32 [B, M, L, H], L2 over d BEFORE aggregation.
    * shift_first_mean, shift_last_mean: float32 [B, H, d], all-layer means.
    * shift_score: float32 [B, H], log1p(L2(last_mean - first_mean)).
    * norm_score: float32 [B, H], mean of the last 5 rounds x deepest 3
      layers of layer_norms, with configurable tail sizes recorded below.
    * hook_counts: int64 [M, L]; step_indices/layer_indices: int64 vectors.
    * scalar metadata: num_steps, num_layers, action_horizon, hidden_size,
      norm_tail_steps, norm_tail_layers, schema_version, capture_branch.

    SR is intentionally not computed here. The scorer uses the last forward's
    action vectors and its chosen adjacent-action window. No data from M3/M5
    side chains is accepted by the default main_steps=10 observer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        action_horizon: int | None = None,
        *,
        main_steps: int = 10,
        norm_tail_steps: int = 5,
        norm_tail_layers: int = 3,
    ) -> None:
        self.main_steps = _positive_int(main_steps, "main_steps")
        self.norm_tail_steps = _positive_int(norm_tail_steps, "norm_tail_steps")
        self.norm_tail_layers = _positive_int(norm_tail_layers, "norm_tail_layers")
        if action_horizon is None:
            action_horizon = getattr(getattr(model, "config", None), "action_horizon", None)
        self.action_horizon = _positive_int(action_horizon, "action_horizon")
        try:
            expert = model.paligemma_with_expert.gemma_expert.model
            layers = tuple(expert.layers)
        except AttributeError as exc:
            raise ValueError(
                "Expected model.paligemma_with_expert.gemma_expert.model.layers"
            ) from exc
        self.num_layers = len(layers)  # normally 18; never hardcoded
        if self.num_layers < self.norm_tail_layers:
            raise ValueError("norm_tail_layers exceeds the runtime expert layer count")
        if self.main_steps < self.norm_tail_steps:
            raise ValueError("norm_tail_steps exceeds main_steps")
        if any(not isinstance(layer, torch.nn.Module) for layer in layers):
            raise TypeError("Every expert layer must be a torch.nn.Module")

        self._closed = False
        self._handles: list[Any] = []
        self._clear_capture()
        try:
            for layer_idx, layer in enumerate(layers):
                self._handles.append(layer.register_forward_hook(self._make_hook(layer_idx)))
        except BaseException:
            self.close()
            raise

    def _clear_capture(self) -> None:
        self._active_idx: int | None = None
        self._next_idx = 0
        self._next_layer = 0
        self._hook_counts = np.zeros((self.main_steps, self.num_layers), dtype=np.int64)
        self._shape: tuple[int, int, int] | None = None
        self._device: torch.device | None = None
        self._layer_norms: torch.Tensor | None = None
        self._last_hidden: torch.Tensor | None = None
        self._first_sum: torch.Tensor | None = None
        self._last_sum: torch.Tensor | None = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SignalObserver is closed; create a new observer")

    def begin_step(self, idx: int, M: int) -> None:
        """Enable capture for exactly one main-chain forward, with zero-based idx."""
        self._require_open()
        if self._active_idx is not None:
            raise RuntimeError("A step is already active; call end_step() first")
        if isinstance(idx, bool):
            raise ValueError("idx must be a zero-based integer")
        idx = operator.index(idx)
        M = _positive_int(M, "M")
        if M != self.main_steps:
            raise ValueError(f"Only the M{self.main_steps} main chain is captured; got M{M}")
        if not 0 <= idx < M or idx != self._next_idx:
            raise ValueError(
                f"Expected main step {self._next_idx}, got {idx}; flush/reset between queries"
            )
        self._next_layer = 0
        self._active_idx = idx

    def end_step(self) -> None:
        """Disable capture and verify that every runtime expert layer fired once."""
        self._require_open()
        idx = self._active_idx
        if idx is None:
            raise RuntimeError("No main step is active")
        self._active_idx = None  # always stop capture, including validation failures
        counts = self._hook_counts[idx]
        if self._next_layer != self.num_layers or not np.all(counts == 1):
            raise RuntimeError(
                f"Incomplete expert hooks at step {idx}: {counts.tolist()}; reset required"
            )
        self._next_idx += 1

    @contextmanager
    def step(self, idx: int, M: int) -> Iterator[SignalObserver]:
        """Preferred scope: exceptions discard the partial query and disable hooks."""
        self.begin_step(idx, M)
        try:
            yield self
            self.end_step()
        except BaseException:
            self._clear_capture()
            raise

    def _make_hook(self, layer_idx: int):
        def capture(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            idx = self._active_idx
            if idx is None or self._closed:
                return None
            if layer_idx != self._next_layer:
                raise RuntimeError(
                    f"Unexpected expert hook at step {idx}: layer {layer_idx}, "
                    f"expected {self._next_layer}; duplicate/side-chain forward?"
                )
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise TypeError("Expected decoder output[0] (or output) with shape [B,T,d]")
            if hidden.shape[1] < self.action_horizon:
                raise ValueError("Expert output has fewer tokens than action_horizon")
            shape = (hidden.shape[0], self.action_horizon, hidden.shape[2])
            if shape[0] == 0 or shape[2] == 0 or not hidden.is_floating_point():
                raise ValueError("Expected a nonempty floating-point expert hidden tensor")
            if self._shape is None:
                self._allocate(shape, hidden.device)
            elif shape != self._shape or hidden.device != self._device:
                raise ValueError("Expert action shape/device changed within one query")

            # Never keep model-owned storage or an autograd graph. Only telemetry
            # buffers are mutated, so even an FP32 no-op cast cannot change output.
            with torch.no_grad():
                action_hidden = hidden.detach()[:, -self.action_horizon :, :].to(torch.float32)
                self._layer_norms[idx, layer_idx].copy_(
                    torch.linalg.vector_norm(action_hidden, ord=2, dim=-1)
                )
                if idx == 0:
                    self._first_sum.add_(action_hidden)
                if idx == self.main_steps - 1:
                    self._last_sum.add_(action_hidden)
                if layer_idx == self.num_layers - 1:
                    self._last_hidden[idx].copy_(action_hidden)
            self._hook_counts[idx, layer_idx] += 1
            self._next_layer += 1
            return None

        return capture

    def _allocate(self, shape: tuple[int, int, int], device: torch.device) -> None:
        self._shape = shape
        self._device = device
        batch, horizon, width = shape
        options = {"dtype": torch.float32, "device": device}
        self._layer_norms = torch.empty(
            (self.main_steps, self.num_layers, batch, horizon), **options
        )
        self._last_hidden = torch.empty((self.main_steps, batch, horizon, width), **options)
        self._first_sum = torch.zeros(shape, **options)
        self._last_sum = torch.zeros(shape, **options)

    def flush(self) -> dict[str, np.ndarray]:
        """Validate a complete query, transfer owned arrays to CPU, then reset.

        No CPU transfer or tensor-value synchronization occurs in a layer hook.
        A failed flush retains the buffers for diagnosis; reset() discards them.
        Returned arrays remain valid after reset(), the next query, or close().
        """
        self._require_open()
        if self._active_idx is not None:
            raise RuntimeError("A step is active; call end_step() before flush()")
        if self._next_idx != self.main_steps or not np.all(self._hook_counts == 1):
            raise RuntimeError(
                f"Cannot flush incomplete main chain: {self._next_idx}/{self.main_steps} steps"
            )

        with torch.no_grad():
            first_mean = self._first_sum / self.num_layers
            last_mean = self._last_sum / self.num_layers
            shift_score = torch.log1p(torch.linalg.vector_norm(last_mean - first_mean, dim=-1))
            norm_score = self._layer_norms[
                -self.norm_tail_steps :, -self.norm_tail_layers :
            ].mean(dim=(0, 1))
            tensors = {
                "last_layer_hidden": self._last_hidden.permute(1, 0, 2, 3),
                "layer_norms": self._layer_norms.permute(2, 0, 1, 3),
                "shift_first_mean": first_mean,
                "shift_last_mean": last_mean,
                "shift_score": shift_score,
                "norm_score": norm_score,
            }
            result = {
                key: tensor.detach().to(device="cpu").contiguous().numpy().copy()
                for key, tensor in tensors.items()
            }
        for key, array in result.items():
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Nonfinite observer telemetry in {key}")
        result.update(
            hook_counts=self._hook_counts.copy(),
            step_indices=np.arange(self.main_steps, dtype=np.int64),
            layer_indices=np.arange(self.num_layers, dtype=np.int64),
            num_steps=np.asarray(self.main_steps, dtype=np.int64),
            num_layers=np.asarray(self.num_layers, dtype=np.int64),
            action_horizon=np.asarray(self.action_horizon, dtype=np.int64),
            hidden_size=np.asarray(self._shape[-1], dtype=np.int64),
            norm_tail_steps=np.asarray(self.norm_tail_steps, dtype=np.int64),
            norm_tail_layers=np.asarray(self.norm_tail_layers, dtype=np.int64),
            schema_version=np.asarray("pi05-hidden-observer-v1"),
            capture_branch=np.asarray("main"),
        )
        self._clear_capture()
        return result

    def finish(self) -> dict[str, np.ndarray]:
        """Alias for flush(); this does not remove hooks."""
        return self.flush()

    def reset(self) -> None:
        """Discard this query and disable capture; installed hooks stay attached."""
        self._require_open()
        self._clear_capture()

    def close(self) -> None:
        """Remove this observer's hooks and release buffers; safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._active_idx = None
        try:
            for handle in self._handles:
                handle.remove()
        finally:
            self._handles.clear()
            self._clear_capture()

    def __enter__(self) -> SignalObserver:
        self._require_open()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()
