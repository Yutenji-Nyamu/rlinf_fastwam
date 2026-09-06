from __future__ import annotations

import json
import sys
import time

import torch

sys.path.insert(0, "/root/autodl-tmp")
from qam_real_model_basic_probe import configure_qam, make_fixed_env_obs  # noqa: E402

from rlinf.algorithms.qam.contracts import embed_planned_adjoint  # noqa: E402
from rlinf.models import get_model  # noqa: E402
from rlinf.models.embodiment.base_policy import ForwardType  # noqa: E402


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def main() -> None:
    torch.manual_seed(20260731)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    cfg = configure_qam()
    print("QAM_REAL_GRADIENT_MODEL_LOAD_BEGIN=1", flush=True)
    model = get_model(cfg.actor.model)
    assert model is not None
    model = model.to(device)
    model.eval()

    conditioning = model(
        forward_type=ForwardType.QAM_FLOW,
        operation="conditioning",
        env_obs=make_fixed_env_obs(),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(20260731)
    noise = torch.randn(
        1,
        50,
        32,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    sample_start = time.perf_counter()
    sampled = model(
        forward_type=ForwardType.QAM_FLOW,
        operation="sample_ode",
        env_obs=make_fixed_env_obs(),
        noise=noise,
        flow_steps=10,
    )
    synchronize(device)
    sample_seconds = time.perf_counter() - sample_start
    raw_active = sampled["qam_raw_endpoint"][:, :20, :14]
    canonical_active = sampled["qam_planned_action_normalized"]
    overflow = (raw_active.abs() > 1.0)
    overflow_fraction = overflow.float().mean().item()
    overflow_count = int(overflow.sum().item())
    max_abs = raw_active.abs().max().item()
    max_excess = (raw_active.abs() - 1.0).clamp_min(0).max().item()
    assert torch.equal(canonical_active, raw_active.clamp(-1.0, 1.0))
    assert torch.equal(sampled["actions"][:, 20:, :], sampled["qam_raw_endpoint"][:, 20:, :])
    assert torch.equal(sampled["actions"][:, :20, 14:], sampled["qam_raw_endpoint"][:, :20, 14:])

    del sampled
    torch.cuda.empty_cache()
    base_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    x_t = noise.detach().clone().requires_grad_(True)
    planned_terminal = torch.randn(
        1,
        20,
        14,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    terminal_adjoint = embed_planned_adjoint(planned_terminal)
    vjp_start = time.perf_counter()
    behavior_velocity = model(
        forward_type=ForwardType.QAM_FLOW,
        operation="velocity",
        state=conditioning["state"],
        x_t=x_t,
        time_qam=torch.tensor([0.625], device=device),
        prefix_pad_masks=conditioning["prefix_pad_masks"],
        past_key_values=conditioning["past_key_values"],
        route="behavior",
    )
    behavior_vjp = torch.autograd.grad(
        (behavior_velocity * terminal_adjoint).sum(),
        x_t,
        retain_graph=False,
        create_graph=False,
    )[0]
    synchronize(device)
    vjp_seconds = time.perf_counter() - vjp_start
    assert torch.isfinite(behavior_vjp).all()
    assert not any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("qam_fine.")
    )
    vjp_peak = torch.cuda.max_memory_allocated(device)
    vjp_inactive_abs_max = max(
        behavior_vjp[:, 20:, :].abs().max().item(),
        behavior_vjp[:, :20, 14:].abs().max().item(),
    )

    del behavior_velocity, behavior_vjp, x_t
    torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)
    model.train()
    fine_base_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    fine_start = time.perf_counter()
    fine_velocity = model(
        forward_type=ForwardType.QAM_FLOW,
        operation="velocity",
        state=conditioning["state"],
        x_t=noise,
        time_qam=torch.tensor([0.625], device=device),
        prefix_pad_masks=conditioning["prefix_pad_masks"],
        past_key_values=conditioning["past_key_values"],
        route="fine",
    )
    fine_loss = fine_velocity.square().mean()
    fine_loss.backward()
    synchronize(device)
    fine_seconds = time.perf_counter() - fine_start
    fine_peak = torch.cuda.max_memory_allocated(device)
    fine_grad_parameters = 0
    fine_grad_finite = True
    fine_grad_norm_sq = torch.zeros((), device=device)
    for name, parameter in model.named_parameters():
        if name.startswith("qam_fine.") and parameter.grad is not None:
            fine_grad_parameters += 1
            fine_grad_finite = fine_grad_finite and bool(
                torch.isfinite(parameter.grad).all().item()
            )
            fine_grad_norm_sq += parameter.grad.float().square().sum()
        elif not name.startswith("qam_fine."):
            assert parameter.grad is None, name
    assert fine_grad_parameters > 0
    assert fine_grad_finite

    result = {
        "flow_steps": 10,
        "sample_seconds": sample_seconds,
        "raw_active_elements": raw_active.numel(),
        "raw_active_overflow_count": overflow_count,
        "raw_active_overflow_fraction": overflow_fraction,
        "raw_active_max_abs": max_abs,
        "raw_active_max_excess": max_excess,
        "vjp_seconds": vjp_seconds,
        "vjp_base_allocated_bytes": base_allocated,
        "vjp_peak_allocated_bytes": vjp_peak,
        "vjp_peak_delta_bytes": vjp_peak - base_allocated,
        "vjp_inactive_abs_max_after_behavior_pullback": vjp_inactive_abs_max,
        "fine_backward_seconds": fine_seconds,
        "fine_backward_base_allocated_bytes": fine_base_allocated,
        "fine_backward_peak_allocated_bytes": fine_peak,
        "fine_backward_peak_delta_bytes": fine_peak - fine_base_allocated,
        "fine_grad_parameter_tensors": fine_grad_parameters,
        "fine_grad_global_norm": fine_grad_norm_sq.sqrt().item(),
        "fine_grad_finite": fine_grad_finite,
    }
    print("QAM_REAL_GRADIENT_RESULT=" + json.dumps(result, sort_keys=True))
    print("QAM_REAL_GRADIENT_OK=1", flush=True)


if __name__ == "__main__":
    main()
