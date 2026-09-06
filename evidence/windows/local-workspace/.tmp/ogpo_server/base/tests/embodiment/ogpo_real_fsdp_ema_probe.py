"""Two-rank real-checkpoint probe for RLinf's OpenPI OGPO EMA under FSDP1."""

from __future__ import annotations

import os
import time

import hydra
import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy


def gib(value: int) -> float:
    return value / 1024**3


def main() -> None:
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"probe requires two ranks, got {world_size}")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    repo = "/root/autodl-tmp/RLinf_ogpo_pi0_robotwin"
    config_dir = f"{repo}/examples/embodiment/config"
    with hydra.initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = hydra.compose(config_name="robotwin_adjust_bottle_ogpo_openpi")
    OmegaConf.resolve(cfg)

    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.openpi import get_model

    try:
        load_started = time.perf_counter()
        module = get_model(cfg.actor.model)
        load_seconds = time.perf_counter() - load_started
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=cfg.actor.fsdp_config,
            is_lora=cfg.actor.model.is_lora,
            model_type=cfg.actor.model.model_type,
        )
        wrap_started = time.perf_counter()
        model = FSDP(
            module=module,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=None,
                reduce_dtype=None,
                buffer_dtype=None,
            ),
            sync_module_states=True,
            forward_prefetch=False,
            limit_all_gathers=False,
            use_orig_params=True,
            cpu_offload=CPUOffload(offload_params=False),
        )
        wrap_seconds = time.perf_counter() - wrap_started

        paligemma_head = (
            model.module.paligemma_with_expert.paligemma.lm_head
        )
        expert_head = (
            model.module.paligemma_with_expert.gemma_expert.lm_head
        )
        if isinstance(paligemma_head, FSDP):
            raise RuntimeError("tied PaliGemma lm_head crossed an FSDP unit")
        if not isinstance(expert_head, FSDP):
            raise RuntimeError("independent action-expert lm_head was not wrapped")

        state_batch = 4
        group_size = 8
        env_obs = {
            "main_images": torch.randint(
                0, 256, (state_batch, 240, 320, 3), dtype=torch.uint8
            ),
            "wrist_images": torch.randint(
                0, 256, (state_batch, 2, 240, 320, 3), dtype=torch.uint8
            ),
            "extra_view_images": None,
            "states": torch.zeros(state_batch, 14, dtype=torch.float32),
            "task_descriptions": ["adjust the bottle"] * state_batch,
        }
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        candidate_started = time.perf_counter()
        candidate_output = model(
            forward_type=ForwardType.OGPO_FLOW,
            operation="actor_batch",
            env_obs=env_obs,
            group_size=group_size,
        )
        candidate_delta = (
            candidate_output["current_chain_score"]
            - candidate_output["old_chain_score"]
        )
        candidate_score_delta = float(candidate_delta.detach().abs().max().item())
        if candidate_score_delta > 1e-5:
            raise RuntimeError(
                f"real FSDP same-chain identity differs: {candidate_score_delta}"
            )
        candidate_output["current_chain_score"].mean().backward()
        actor_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not actor_gradients or not all(
            torch.isfinite(gradient).all() for gradient in actor_gradients
        ):
            raise RuntimeError("real FSDP candidate microbatch produced invalid gradients")
        candidate_seconds = time.perf_counter() - candidate_started
        candidate_peak_allocated = gib(torch.cuda.max_memory_allocated())
        candidate_peak_reserved = gib(torch.cuda.max_memory_reserved())
        candidate_grad_tensors = len(actor_gradients)
        del candidate_output, candidate_delta, actor_gradients, env_obs
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        local_pairs = list(
            model.module.ogpo_target._named_parameter_pairs(model.module)
        )
        if not local_pairs:
            raise RuntimeError("real FSDP model found no EMA parameter pairs")
        old_targets = {
            name: target.detach().float().clone()
            for name, target, _ in local_pairs
        }
        with torch.no_grad():
            for _, _, source in local_pairs:
                source.add_(torch.full_like(source, 0.125))

        model(
            forward_type=ForwardType.OGPO_FLOW,
            operation="ema_update",
            tau=0.25,
        )
        updated_pairs = list(
            model.module.ogpo_target._named_parameter_pairs(model.module)
        )
        max_update_error = 0.0
        for name, target, _ in updated_pairs:
            if target.numel():
                expected = old_targets[name] + 0.03125
                max_update_error = max(
                    max_update_error,
                    float((target.float() - expected).abs().max().item()),
                )
        if max_update_error > 0.01:
            raise RuntimeError(f"real visible EMA update differs: {max_update_error}")

        shadow = model(
            forward_type=ForwardType.OGPO_FLOW,
            operation="ema_shadow_state",
        )
        if not shadow or any(value.dtype != torch.float32 for value in shadow.values()):
            raise RuntimeError("real FSDP model did not create an FP32 EMA shadow")
        snapshot_path = f"/tmp/ogpo_real_fsdp_ema_shadow_rank_{rank}.pt"
        torch.save(shadow, snapshot_path)
        restored = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        model(
            forward_type=ForwardType.OGPO_FLOW,
            operation="load_ema_shadow_state",
            state={},
        )
        model(
            forward_type=ForwardType.OGPO_FLOW,
            operation="load_ema_shadow_state",
            state=restored,
        )
        roundtrip = model(
            forward_type=ForwardType.OGPO_FLOW,
            operation="ema_shadow_state",
        )
        if roundtrip.keys() != shadow.keys():
            raise RuntimeError("real EMA shadow keys changed after round-trip")
        for name, value in shadow.items():
            torch.testing.assert_close(roundtrip[name], value)

        local_result = {
            "rank": rank,
            "load_seconds": round(load_seconds, 2),
            "wrap_seconds": round(wrap_seconds, 2),
            "pairs": len(local_pairs),
            "nonempty_pairs": sum(target.numel() > 0 for _, target, _ in local_pairs),
            "shadow_tensors": len(shadow),
            "shadow_elements": sum(value.numel() for value in shadow.values()),
            "max_update_error": max_update_error,
            "peak_allocated_gib": round(gib(torch.cuda.max_memory_allocated()), 3),
            "peak_reserved_gib": round(gib(torch.cuda.max_memory_reserved()), 3),
            "candidate_states": state_batch,
            "candidate_group": group_size,
            "candidate_seconds": round(candidate_seconds, 2),
            "candidate_score_delta": candidate_score_delta,
            "candidate_grad_tensors": candidate_grad_tensors,
            "candidate_peak_allocated_gib": round(candidate_peak_allocated, 3),
            "candidate_peak_reserved_gib": round(candidate_peak_reserved, 3),
        }
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_result)
        if rank == 0:
            print(f"REAL_FSDP_EMA_PROBE_OK results={gathered}", flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
