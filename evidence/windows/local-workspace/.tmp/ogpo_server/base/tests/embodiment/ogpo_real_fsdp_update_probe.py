"""Two-rank real-checkpoint probe of one production OGPO actor/Q update."""

from __future__ import annotations

import os
import time

import hydra
import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


class _ProbeLogger:
    @staticmethod
    def warning(message: str) -> None:
        print(f"PROBE_WARNING {message}", flush=True)


def _observation(batch_size: int) -> dict[str, torch.Tensor]:
    prompt = torch.tensor(list(b"adjust the bottle"), dtype=torch.uint8)
    return {
        "main_images": torch.randint(
            0, 256, (batch_size, 240, 320, 3), dtype=torch.uint8
        ),
        "wrist_images": torch.randint(
            0, 256, (batch_size, 2, 240, 320, 3), dtype=torch.uint8
        ),
        "states": torch.zeros(batch_size, 14, dtype=torch.float32),
        "prompt_utf8": prompt[None].expand(batch_size, -1).clone(),
        "prompt_length": torch.full(
            (batch_size,), prompt.numel(), dtype=torch.long
        ),
    }


def _sequence(batch_size: int):
    from rlinf.data.ogpo_replay import OGPOSequenceBatch

    observation = _observation(batch_size)
    next_observation = {
        name: value.clone() for name, value in observation.items()
    }
    action_model = torch.randn(batch_size, 10, 32).clamp_(-1.0, 1.0)
    action = action_model[:, :, :14].clone()
    rewards = torch.zeros(batch_size, 10, dtype=torch.float32)
    rewards[:, -1] = 1.0
    return OGPOSequenceBatch(
        observation=observation,
        next_observation=next_observation,
        action_model=action_model,
        action=action,
        rewards=rewards,
        terminated=torch.zeros(batch_size, 10, dtype=torch.bool),
        truncated=torch.zeros(batch_size, 10, dtype=torch.bool),
        valid=torch.ones(batch_size, 10, dtype=torch.bool),
        h=torch.full((batch_size,), 10, dtype=torch.long),
        bootstrap_mask=torch.ones(batch_size, dtype=torch.float32),
        row_ids=torch.arange(batch_size * 10, dtype=torch.long).reshape(
            batch_size, 10
        ),
        episode_ids=torch.arange(batch_size, dtype=torch.long),
        start_step_ids=torch.zeros(batch_size, dtype=torch.long),
    )


def _success(sequence):
    from rlinf.data.ogpo_replay import OGPOSuccessBatch

    return OGPOSuccessBatch(
        observation=sequence.observation,
        action_model=sequence.action_model,
        action=sequence.action,
        row_ids=sequence.row_ids,
        episode_ids=sequence.episode_ids,
        start_step_ids=sequence.start_step_ids,
    )


def main() -> None:
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"probe requires two ranks, got {world_size}")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    repo = "/root/autodl-tmp/RLinf_ogpo_pi0_robotwin"
    config_dir = f"{repo}/examples/embodiment/config"
    with hydra.initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = hydra.compose(config_name="robotwin_adjust_bottle_ogpo_openpi")
    OmegaConf.resolve(cfg)

    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.utils.utils import warmup_optimizer_state
    from rlinf.workers.actor.fsdp_ogpo_policy_worker import (
        EmbodiedOGPOFSDPPolicy,
    )

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

        worker = object.__new__(EmbodiedOGPOFSDPPolicy)
        worker.cfg = cfg
        worker.ogpo_cfg = cfg.algorithm.ogpo
        worker.model = model
        worker.device = device
        worker._rank = rank
        worker._world_size = world_size
        worker.critic = None
        worker.target_critic = None
        worker.critic_optimizer = None
        worker.critic_feature_dim = None
        worker.actor_updates = 0
        worker.critic_updates = 0
        worker.policy_version = 0
        worker.optimizer_steps = 0
        worker.critic_warmup_steps = 0
        worker._logger = _ProbeLogger()
        worker._strategy = FSDPStrategy(
            cfg=cfg.actor,
            world_size=world_size,
            dp_group=None,
            logger=worker._logger,
        )
        worker.optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(cfg.actor.optim.lr),
            betas=(
                float(cfg.actor.optim.adam_beta1),
                float(cfg.actor.optim.adam_beta2),
            ),
            eps=float(cfg.actor.optim.adam_eps),
            weight_decay=float(cfg.actor.optim.weight_decay),
        )
        warmup_optimizer_state(worker.optimizer)
        worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            worker.optimizer, lambda _: 1.0
        )
        worker.grad_scaler = ShardedGradScaler(enabled=False)

        local_batch = int(cfg.algorithm.ogpo.state_batch_size) // world_size
        sequence = _sequence(local_batch)
        success = _success(sequence) if rank == 0 else None
        model.train()
        torch.cuda.reset_peak_memory_stats(device)

        torch.cuda.synchronize(device)
        target_started = time.perf_counter()
        target_next = worker._target_action(
            worker._batch_to_env_obs(sequence.next_observation)
        )
        torch.cuda.synchronize(device)
        target_seconds = time.perf_counter() - target_started

        actor_started = time.perf_counter()
        actor_metrics = worker._actor_update(sequence, success)
        torch.cuda.synchronize(device)
        actor_seconds = time.perf_counter() - actor_started

        critic_started = time.perf_counter()
        critic_metrics = worker._critic_update(sequence, target_next)
        torch.cuda.synchronize(device)
        critic_seconds = time.perf_counter() - critic_started

        metrics = {**actor_metrics, **critic_metrics}
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise RuntimeError(f"non-finite full-update metric: {metrics}")
        if (worker.actor_updates, worker.critic_updates, worker.policy_version) != (
            1,
            1,
            1,
        ):
            raise RuntimeError("production update counters did not advance once")
        if worker.optimizer_steps != 1:
            raise RuntimeError("actor optimizer did not step exactly once")
        if worker.critic_optimizer is None:
            raise RuntimeError("critic optimizer was not initialized")
        if worker.critic_optimizer.param_groups[0]["weight_decay"] != float(
            cfg.algorithm.ogpo.critic_weight_decay
        ):
            raise RuntimeError("critic optimizer weight decay differs from config")

        local_result = {
            "rank": rank,
            "load_seconds": round(load_seconds, 2),
            "target_seconds": round(target_seconds, 2),
            "actor_seconds": round(actor_seconds, 2),
            "critic_seconds": round(critic_seconds, 2),
            "success_bc": rank == 0,
            "actor_loss": actor_metrics["ogpo/actor_loss"],
            "bc_loss": actor_metrics["ogpo/bc_loss"],
            "ratio": actor_metrics["ogpo/ratio"],
            "critic_loss": critic_metrics["ogpo/critic_loss"],
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated(device) / 1024**3, 3
            ),
            "peak_reserved_gib": round(
                torch.cuda.max_memory_reserved(device) / 1024**3, 3
            ),
        }
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_result)
        if rank == 0:
            print(f"REAL_FSDP_FULL_UPDATE_OK results={gathered}", flush=True)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
