"""Detached action-expert residual L2 from the existing ODE10 forwards."""

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
    hook_maps = [
        (
            layer,
            layer._forward_hooks,
            layer._forward_hooks_with_kwargs,
            layer._forward_hooks_always_called,
        )
        for layer in layers[-3:]
    ]
    for layer, hooks, kwargs_hooks, always_hooks in hook_maps:
        layer._forward_hooks = hooks.copy()
        layer._forward_hooks_with_kwargs = kwargs_hooks.copy()
        layer._forward_hooks_always_called = always_hooks.copy()

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
        for layer, hooks, kwargs_hooks, always_hooks in hook_maps:
            layer._forward_hooks = hooks
            layer._forward_hooks_with_kwargs = kwargs_hooks
            layer._forward_hooks_always_called = always_hooks


def reduce_expert_norm(values):
    if len(values) != 15:
        raise ValueError("Norm requires exactly five steps times three layers")
    result = torch.stack(values).mean(dim=0).detach().float()
    if result.ndim != 2 or not torch.isfinite(result).all() or (result < 0).any():
        raise ValueError("Norm must be finite nonnegative [B,H]")
    return result.contiguous()
