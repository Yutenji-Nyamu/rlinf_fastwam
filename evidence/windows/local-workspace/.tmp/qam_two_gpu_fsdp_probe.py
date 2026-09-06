from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

from rlinf.algorithms.qam.core import (
    AMPath,
    adjoint_matching_step_loss,
    clone_parameter_snapshot,
    ema_from_preupdate_,
    ensemble_critic_mse,
    q_chunk_td_target,
    reverse_behavior_adjoint,
)
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.qam_critic import QAMCriticEnsemble
from rlinf.utils.utils import collect_param_names_need_sync


REPO = Path("/root/autodl-tmp/RLinf_qam_pi0_robotwin")
CONFIG_DIR = REPO / "examples" / "embodiment" / "config"


def make_fixed_env_obs(rank: int, batch: int) -> dict:
    height = 480
    width = 640
    pixel_count = height * width * 3
    single = (
        torch.arange(pixel_count, dtype=torch.int64)
        .add(rank * 11)
        .remainder(256)
        .to(torch.uint8)
        .reshape(1, height, width, 3)
    )
    base = single.repeat(batch, 1, 1, 1)
    left = torch.roll(base, shifts=37, dims=2)
    right = torch.flip(base, dims=(1,))
    states = torch.linspace(
        -0.25 + rank * 0.01,
        0.25 + rank * 0.01,
        14,
        dtype=torch.float32,
    ).reshape(1, 14).repeat(batch, 1)
    return {
        "main_images": base,
        "wrist_images": torch.stack((left, right), dim=1),
        "extra_view_images": None,
        "states": states,
        "task_descriptions": ["adjust the bottle"] * batch,
    }


def gather_scalar(value: torch.Tensor) -> list[float]:
    output = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(output, value)
    return [float(item.item()) for item in output]


def module_checksum(module: torch.nn.Module, device: torch.device) -> torch.Tensor:
    checksum = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in module.parameters():
        checksum += parameter.detach().to(torch.float64).sum()
    return checksum


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError(f"probe requires world_size=2, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    batch_size = int(os.environ.get("QAM_PROBE_BATCH", "1"))
    if batch_size <= 0:
        raise ValueError(f"QAM_PROBE_BATCH must be positive, got {batch_size}")
    torch.manual_seed(20260731)
    torch.cuda.reset_peak_memory_stats(device)

    with initialize_config_dir(
        version_base="1.1",
        config_dir=str(CONFIG_DIR),
    ):
        cfg = compose(config_name="robotwin_adjust_bottle_qam_openpi")

    load_start = time.perf_counter()
    model = get_model(cfg.actor.model)
    if model is None:
        raise RuntimeError("QAM model provider returned None")
    load_seconds = time.perf_counter() - load_start

    trainable_names = collect_param_names_need_sync(model)
    if not trainable_names or not all(
        name.startswith("qam_fine.") for name in trainable_names
    ):
        raise AssertionError("only qam_fine parameters may be trainable")

    auto_wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=cfg.actor.fsdp_config,
        is_lora=bool(cfg.actor.model.is_lora),
        model_type=str(cfg.actor.model.model_type),
    )
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=None,
            reduce_dtype=None,
            buffer_dtype=None,
        ),
        sync_module_states=True,
        use_orig_params=True,
        forward_prefetch=False,
        limit_all_gathers=False,
    )
    wrapped_root = fsdp_model.module
    if isinstance(
        wrapped_root.paligemma_with_expert.paligemma.lm_head,
        FSDP,
    ):
        raise AssertionError("tied PaliGemma lm_head crossed an FSDP boundary")
    if not isinstance(
        wrapped_root.paligemma_with_expert.gemma_expert.lm_head,
        FSDP,
    ):
        raise AssertionError("independent expert lm_head lost its legacy wrap")
    fsdp_model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in fsdp_model.parameters() if parameter.requires_grad),
        lr=2.0e-5,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )

    conditioning = fsdp_model(
        forward_type=ForwardType.QAM_FLOW,
        operation="conditioning",
        env_obs=make_fixed_env_obs(rank, batch_size),
    )
    if conditioning["critic_feature"].shape != (batch_size, 4, 2048):
        raise AssertionError(conditioning["critic_feature"].shape)
    if conditioning["block_lengths"] != (256, 256, 256, 48):
        raise AssertionError(conditioning["block_lengths"])

    generator = torch.Generator(device=device)
    generator.manual_seed(9000 + rank)
    flow_steps = 10
    flat_dim = 50 * 32
    path_states = torch.randn(
        flow_steps,
        batch_size,
        flat_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    path_times = (
        torch.arange(flow_steps, device=device, dtype=torch.float32)
        .div(flow_steps)
        .reshape(flow_steps, 1, 1)
        .repeat(1, batch_size, 1)
    )
    path_sigmas = torch.sqrt(
        2.0 * (1.0 - path_times + 1.0 / flow_steps)
        / (path_times + 1.0 / flow_steps)
    )
    terminal = torch.randn(
        batch_size,
        flat_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    def behavior_velocity(
        flat_state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        result = fsdp_model(
            forward_type=ForwardType.QAM_FLOW,
            operation="velocity",
            state=conditioning["state"],
            x_t=flat_state.reshape(flat_state.shape[0], 50, 32),
            time_qam=time.reshape(flat_state.shape[0]),
            prefix_pad_masks=conditioning["prefix_pad_masks"],
            past_key_values=conditioning["past_key_values"],
            route="behavior",
        )
        return result.reshape_as(flat_state)

    path = AMPath(
        states=path_states,
        times=path_times,
        sigmas=path_sigmas,
        endpoint=torch.zeros(batch_size, flat_dim, device=device),
    )
    behavior_adjoints = reverse_behavior_adjoint(
        behavior_velocity,
        path,
        terminal,
        use_backward_vjp=True,
    )
    if not torch.isfinite(behavior_adjoints).all():
        raise FloatingPointError("behavior input VJP is non-finite")
    if any(
        parameter.grad is not None
        for name, parameter in fsdp_model.named_parameters()
        if "qam_fine." not in name
    ):
        raise AssertionError("behavior VJP populated frozen parameter gradients")

    optimizer.zero_grad(set_to_none=True)

    def fine_velocity(
        flat_state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        result = fsdp_model(
            forward_type=ForwardType.QAM_FLOW,
            operation="velocity",
            state=conditioning["state"],
            x_t=flat_state.reshape(flat_state.shape[0], 50, 32),
            time_qam=time.reshape(flat_state.shape[0]),
            prefix_pad_masks=conditioning["prefix_pad_masks"],
            past_key_values=conditioning["past_key_values"],
            route="fine",
        )
        return result.reshape_as(flat_state)

    fine_loss = torch.zeros((), device=device, dtype=torch.float32)
    for index in range(flow_steps):
        contribution = adjoint_matching_step_loss(
            fine_velocity,
            behavior_velocity,
            path.states[index],
            path.times[index],
            path.sigmas[index],
            behavior_adjoints[index],
        )
        contribution.backward()
        fine_loss = fine_loss + contribution.detach()

    local_fine_grad_tensors = 0
    local_fine_grad_finite = True
    for name, parameter in fsdp_model.named_parameters():
        if "qam_fine." in name and parameter.grad is not None:
            local_fine_grad_tensors += 1
            local_fine_grad_finite = local_fine_grad_finite and bool(
                torch.isfinite(parameter.grad).all().item()
            )
        elif "qam_fine." not in name and parameter.grad is not None:
            raise AssertionError(f"frozen parameter received gradient: {name}")
    finite_flag = torch.tensor(
        int(local_fine_grad_finite and local_fine_grad_tensors > 0),
        device=device,
    )
    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
    if not bool(finite_flag.item()):
        raise FloatingPointError("F1 sharded gradients are missing or non-finite")
    strategy = FSDPStrategy(
        cfg.actor,
        world_size,
        dp_group=dist.group.WORLD,
    )
    fine_grad_norm = strategy.clip_grad_norm_(fsdp_model)
    grad_norms = gather_scalar(
        torch.tensor(fine_grad_norm, device=device, dtype=torch.float64)
    )
    if max(grad_norms) - min(grad_norms) > 1e-6:
        raise AssertionError(f"FSDP grad norms disagree: {grad_norms}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.manual_seed(2726)
    critic = QAMCriticEnsemble(
        feature_dim=conditioning["critic_feature"].shape[-1],
        num_q_heads=10,
        hidden_dims=(512, 512, 512, 512),
    ).to(device=device, dtype=torch.float32)
    for parameter in critic.parameters():
        dist.broadcast(parameter.data, src=0)
    target_critic = copy.deepcopy(critic).eval().requires_grad_(False)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=3.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    feature = conditioning["critic_feature"].detach().to(torch.float32)
    proprio = conditioning["state"][:, :14].detach().to(torch.float32)
    action = torch.tanh(
        torch.randn(
            batch_size,
            20,
            14,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    )
    next_action = torch.tanh(
        torch.randn(
            batch_size,
            20,
            14,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    )
    with torch.no_grad():
        next_q = target_critic(feature, proprio, next_action)
        td_target = q_chunk_td_target(
            torch.full((batch_size,), float(rank), device=device),
            torch.ones(batch_size, device=device),
            next_q,
            discount_h=0.99**20,
            rho=0.5,
        )
    preupdate = clone_parameter_snapshot(critic)
    q_values = critic(feature, proprio, action)
    critic_loss = ensemble_critic_mse(
        q_values,
        td_target,
        torch.ones(batch_size, device=device, dtype=torch.bool),
    )
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    for parameter in critic.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    if not torch.isfinite(critic_grad_norm):
        raise FloatingPointError("critic gradient norm is non-finite")
    critic_optimizer.step()
    ema_from_preupdate_(target_critic, preupdate, tau=0.005)

    critic_checksums = gather_scalar(module_checksum(critic, device))
    target_checksums = gather_scalar(module_checksum(target_critic, device))
    if max(critic_checksums) - min(critic_checksums) > 1e-8:
        raise AssertionError(f"critic ranks diverged: {critic_checksums}")
    if max(target_checksums) - min(target_checksums) > 1e-8:
        raise AssertionError(f"target critic ranks diverged: {target_checksums}")

    torch.cuda.synchronize(device)
    peak = torch.tensor(
        torch.cuda.max_memory_allocated(device),
        device=device,
        dtype=torch.float64,
    )
    peaks = gather_scalar(peak)
    loads = gather_scalar(torch.tensor(load_seconds, device=device, dtype=torch.float64))
    fine_losses = gather_scalar(fine_loss.detach().to(torch.float64))
    critic_losses = gather_scalar(critic_loss.detach().to(torch.float64))

    if rank == 0:
        print(
            "QAM_TWO_GPU_FSDP_RESULT="
            + json.dumps(
                {
                    "world_size": world_size,
                    "batch_size_per_rank": batch_size,
                    "load_seconds": loads,
                    "trainable_name_count": len(trainable_names),
                    "fine_losses": fine_losses,
                    "fine_grad_norms": grad_norms,
                    "critic_losses": critic_losses,
                    "critic_checksums": critic_checksums,
                    "target_checksums": target_checksums,
                    "peak_allocated_bytes": peaks,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print("QAM_TWO_GPU_FSDP_OK=1", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
