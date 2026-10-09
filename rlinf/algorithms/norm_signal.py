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

"""Detached action-expert residual L2 from existing ODE4 or ODE10 forwards."""

from contextlib import contextmanager

import torch

NORM_SIGNAL_SPEC = {
    "version": 1,
    "kind": "norm_residual_t5_l3",
    "steps": [10],
    "tail_steps": 5,
    "deep_layers": 3,
    "action_dim": 14,
    "readout": "post_residual_pre_final_norm",
    "reduction": "mean_of_l2",
}


@contextmanager
def capture_expert_norm(model, horizon):
    layers = tuple(model.paligemma_with_expert.gemma_expert.model.layers)
    if len(layers) < 3 or horizon <= 0:
        raise ValueError("Norm requires three expert blocks and a positive horizon")
    values, handles = [], []

    def hook(index):
        def collect(_module, _args, output):
            if len(values) != index:
                raise ValueError("Missing, repeated or out-of-order Norm block")
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 3
                or hidden.shape[1] < horizon
            ):
                raise ValueError("Norm requires residual [B,T,D] expert output")
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
            raise ValueError("Norm did not observe all three residual blocks")
    finally:
        for handle in handles:
            handle.remove()


def reduce_expert_norm(values, *, tail_steps=5):
    if (
        type(tail_steps) is not int
        or tail_steps not in (4, 5)
        or len(values) != tail_steps * 3
    ):
        raise ValueError("Norm requires exactly tail_steps times three layers")
    result = torch.stack(values).mean(dim=0).detach().float()
    if result.ndim != 2 or not torch.isfinite(result).all() or (result < 0).any():
        raise ValueError("Norm must be finite nonnegative [B,H]")
    return result.contiguous()
