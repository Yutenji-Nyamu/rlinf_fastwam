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

"""Native pi05 preprocessing and trainable action-expert checkpoint overlays."""

import torch
from torch import nn

PREFIXES = (
    "paligemma_with_expert.gemma_expert.model.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
    "distill_stop_proj.",
)


def expert_state(model: nn.Module) -> dict:
    """Only the mutable expert; the pinned, frozen base remains a dependency."""
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if n.startswith(PREFIXES)
    }


def load_expert(model: nn.Module, state: dict) -> None:
    """Require every expert tensor, rejecting missing or unrelated parameters."""
    params = {n: p for n, p in model.named_parameters() if n.startswith(PREFIXES)}
    if params.keys() != state.keys():
        raise ValueError(f"Expert key mismatch: {params.keys() ^ state.keys()}")
    with torch.no_grad():
        for name, parameter in params.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f"Expert shape mismatch: {name}")
            parameter.copy_(state[name])


class Velocity(nn.Module):
    """Expose differentiable native velocity with FP32 trainable master weights."""

    def __init__(self, model: nn.Module, trainable: bool):
        super().__init__()
        self.model = model
        for name, parameter in model.named_parameters():
            selected = name.startswith(PREFIXES)
            parameter.requires_grad_(selected and trainable)
            if selected:
                parameter.data = parameter.data.float()
        model.eval()

    def forward(self, x, t, context, stop=None):
        """Evaluate the action expert using the shared frozen observation cache."""
        state, masks, cache = context
        return self.model.get_velocity(state, x, t, masks, cache, stop_time=stop)[0]

    def parameters_to_train(self):
        """Return the action-expert parameters used by the optimizer."""
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def endpoint(self, noise, context, steps: int = 10):
        """Use the accepted native eval ODE, including its exact time grid."""
        state, masks, cache = context
        return self.model._sample_actions_with_prefix_cache(
            state,
            None,
            masks,
            cache,
            noise=noise,
            mode="eval",
            compute_values=False,
            num_steps=steps,
        )["actions"]


@torch.no_grad()
def prepare(model, obs, actions):
    """Reuse accepted EXPO/native RGB, absolute-action normalization and KV path."""
    from openpi.models.model import Observation

    packed = model.obs_processor(dict(obs, extra_view_images=None))
    packed["actions"] = actions.cpu().float()
    processed = model.precision_processor(
        model.input_transform(packed, transpose=False)
    )
    images, masks, tokens, token_masks, state = model._preprocess_observation(
        Observation.from_dict(processed),
        train=False,
    )
    _, prefix_masks, cache = model._build_prefix_cache(
        images, masks, tokens, token_masks
    )
    return processed["actions"].float(), (state, prefix_masks, cache)


@torch.no_grad()
def prepare_fql(model, obs):
    """Native observation processing, sharing one VLM pass with actor and Q."""
    from openpi.models.model import Observation

    packed = model.obs_processor(dict(obs, extra_view_images=None))
    processed = model.precision_processor(
        model.input_transform(packed, transpose=False)
    )
    images, masks, tokens, token_masks, state = model._preprocess_observation(
        Observation.from_dict(processed), train=False
    )
    prefix, valid, cache = model._build_prefix_cache(images, masks, tokens, token_masks)
    from .q_evaluation import pooled_features

    features = pooled_features(prefix, valid, state)
    return (state, valid, cache), features
