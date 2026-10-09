# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Current-policy signals for the existing GRPO action-advantage mapper."""

import math
from contextlib import contextmanager

import torch


def grpo_signal_contract(config: dict) -> dict:
    """Return an explicit signal identity; legacy DVCA keeps its old contract."""
    source = config.get("signal_source", "dvca")
    if source == "dvca":
        return {}
    if source not in {"ugrow_10_5", "norm_tail5_layers3"}:
        raise ValueError("Unknown GRPO signal_source.")
    if (
        config.get("mode") not in {"observe", "apply"}
        or config.get("normalization") != "two_level_group"
        or config.get("application") != "chunk_clipped_action_advantage"
    ):
        raise ValueError(
            "GRPO U/Norm requires enabled two-level action-advantage weighting."
        )
    spec = {
        "version": 1,
        "source": source,
        "policy": "current_rollout_policy",
        "horizon": 50,
        "action_dim": 14,
        "coordinate_space": "normalized",
    }
    if source == "ugrow_10_5":
        spec.update(
            solver="flow_ode",
            steps=[10, 5],
            initial_noise="same_full_main_noise",
            reduction="population_std_over_rms_then_mean_coordinates",
            epsilon=1e-8,
        )
    else:
        spec.update(
            chain="native_train_sampler",
            tail_steps=5,
            tail_layers=3,
            readout="expert_block_output_before_final_norm",
            reduction="mean_of_per_token_l2",
        )
    return spec


def grpo_signal_key(config: dict) -> str:
    """Keep legacy tensor keys while naming the new signal independently."""
    if config.get("signal_source", "dvca") == "dvca":
        return f"dvac_v_l{int(config.get('selected_l', 3))}"
    return "grpo_action_signal"


def compute_ugrow_signal(
    main_actions: torch.Tensor,
    comparison_actions: torch.Tensor,
    action_dim: int = 14,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return detached float32 disagreement [B,H] over valid action coordinates.

    Both arguments are complete normalized action chunks from the same input
    and initial noise. The numerator is the population standard deviation of
    two predictions, abs(a / 2 - b / 2); the denominator is their RMS plus eps.
    Float64 hypot avoids intermediate overflow without clipping the signal.
    """
    if (
        not isinstance(action_dim, int)
        or isinstance(action_dim, bool)
        or action_dim <= 0
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("U-GROW requires a positive action_dim and finite eps > 0.")
    if (
        not isinstance(main_actions, torch.Tensor)
        or not isinstance(comparison_actions, torch.Tensor)
        or main_actions.ndim != 3
        or main_actions.shape != comparison_actions.shape
        or main_actions.shape[-1] < action_dim
        or main_actions.shape[1] == 0
        or main_actions.device != comparison_actions.device
        or not torch.is_floating_point(main_actions)
        or not torch.is_floating_point(comparison_actions)
    ):
        raise ValueError("U-GROW requires matching floating [B,H,D] action tensors.")
    a = main_actions.detach()[..., :action_dim].double()
    b = comparison_actions.detach()[..., :action_dim].double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("U-GROW valid action coordinates must be finite.")
    numerator = (a / 2.0 - b / 2.0).abs()
    denominator = torch.hypot(a / math.sqrt(2.0), b / math.sqrt(2.0)) + eps
    result = (numerator / denominator).mean(dim=-1).float()
    if not torch.isfinite(result).all():
        raise ValueError("U-GROW produced a non-finite signal.")
    return result.detach()


@contextmanager
def capture_expert_norm(model, horizon: int):
    """Capture one forward's deepest three pre-final-norm action vectors.

    The list receives exactly three detached FP32 [B,H] norms in layer order.
    Hooks read decoder block outputs, as in the verified inference observer.
    No model output, cache, random state, or persistent model state is changed.
    """
    layers = tuple(model.paligemma_with_expert.gemma_expert.model.layers)
    if len(layers) < 3:
        raise ValueError("Norm signal needs at least three action-expert layers.")
    values, handles = [], []

    def hook(index):
        def collect(_module, _args, output):
            if len(values) != index:
                raise ValueError("Missing, repeated or out-of-order Norm hook.")
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 3
                or hidden.shape[1] < horizon
            ):
                raise ValueError("Norm readout requires expert residual [B,T,D].")
            with torch.no_grad():
                values.append(
                    torch.linalg.vector_norm(
                        hidden.detach()[:, -horizon:].float(), dim=-1
                    )
                )

        return collect

    try:
        for index, layer in enumerate(layers[-3:]):
            handles.append(layer.register_forward_hook(hook(index)))
        yield values
        if len(values) != 3:
            raise ValueError("Norm capture did not observe all three expert blocks.")
    finally:
        for handle in handles:
            handle.remove()
