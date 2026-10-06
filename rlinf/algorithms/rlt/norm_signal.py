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


"""Observe the existing action-expert residual outputs without modifying them."""

from contextlib import contextmanager

import torch


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
