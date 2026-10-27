# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Offline fixed-prior FQL; native pi05 expert, independent shared-task twin Q.

No environment is created here. Requires complete real transition files.
GPU admission is external; CUDA_VISIBLE_DEVICES must equal FQL_GPU_PAIR.
"""

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from train import atomic_torch, save_json

from rlinf.algorithms.pi05_distillation.checkpoint_retention import prune_checkpoints
from rlinf.algorithms.pi05_distillation.fql import (
    TwinQ,
    actor_objective,
    bellman_target,
    prefix_mask,
    update_target,
)
from rlinf.algorithms.pi05_distillation.model import (
    Velocity,
    expert_state,
    load_expert,
    prepare_fql,
)
from rlinf.algorithms.pi05_distillation.transitions import Transitions
from rlinf.models.embodiment.openpi import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--check-export", action="store_true")
    parser.add_argument("--capacity-only", action="store_true")
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    assert c["method"] == "fql_fixed_teacher" and c["teacher_steps"] in (5, 10)
    assert c["execute_chunk"] in (10, 50) and c["horizon"] == 50
    assert c["budget_resolved"], (
        "Run fql_index --config after real collection before training"
    )
    rank, world = (
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )
    assert world in (1, 2)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["FQL_GPU_PAIR"]
    assert len(os.environ["FQL_GPU_PAIR"].split(",")) == world
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.enable_cudnn_sdp(False)
    if world > 1:
        dist.init_process_group("nccl")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(c["seed"] + rank)
    np.random.seed(c["seed"] + rank)
    torch.manual_seed(c["seed"] + rank)
    data = Transitions(c["transition_manifest"])
    if c.get("single_task"):
        data.validate_coordinates(c["model"]["model_path"], c["execute_chunk"], 200)
    batch, micro = c["batch_size"], c["microbatch_per_rank"]
    assert batch % (world * micro) == 0
    accum = batch // (world * micro)
    per_epoch = math.ceil(len(data) / batch)
    total = c["epochs"] * per_epoch
    stop = min(args.stop_step or total, total)
    cfg = copy.deepcopy(c["model"])
    cfg["openpi"].update(
        distill_target_time=False,
        noise_method="flow_ode",
        use_rlt=False,
        use_dsrl=False,
        is_nft=False,
        rtc_enabled=False,
    )
    teacher_model = get_model(OmegaConf.create(cfg))
    student_model = copy.deepcopy(teacher_model)
    student_model.paligemma_with_expert.paligemma = (
        teacher_model.paligemma_with_expert.paligemma
    )
    student_model.paligemma_with_expert.gemma_expert.lm_head = (
        teacher_model.paligemma_with_expert.gemma_expert.lm_head
    )
    teacher = Velocity(teacher_model, False).cuda(rank)
    student = Velocity(student_model, True).cuda(rank)
    sample = data.batch([0])
    sample_obs = sample["obs"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        context, features = prepare_fql(teacher.model, sample_obs)
    q = TwinQ(
        features.shape[-1],
        chunk=c["execute_chunk"],
        width=c["q_width"],
        depth=c["q_depth"],
    ).cuda(rank)
    del context, features
    params = student.parameters_to_train()
    opt = torch.optim.AdamW(
        params,
        lr=c["lr"],
        betas=tuple(c["betas"]),
        eps=1e-8,
        weight_decay=c["weight_decay"],
    )
    opt_q = torch.optim.Adam(q.parameters(), lr=c["q_lr"])
    # DDP broadcasts rank0 initialization before copying the target Q.
    student_run = (
        DDP(student, device_ids=[rank], broadcast_buffers=False)
        if world > 1
        else student
    )
    q_run = DDP(q, device_ids=[rank], broadcast_buffers=False) if world > 1 else q
    target_q = copy.deepcopy(q).requires_grad_(False)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            ck["config"] != c
            or ck["manifest"] != data.manifest
            or ck["world_size"] != world
        ):
            raise ValueError("Full-resume config/data/world-size mismatch")
        load_expert(student.model, ck["student"])
        q.load_state_dict(ck["q"])
        target_q.load_state_dict(ck["target_q"])
        opt.load_state_dict(ck["optimizer"])
        opt_q.load_state_dict(ck["q_optimizer"])
        start = ck["step"]
        rng = ck["rng"][rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], rank)
        del ck
    if rank == 0:
        save_json(
            out / "resolved.json",
            {
                "config": c,
                "transitions": len(data),
                "tasks": data.names,
                "steps_per_epoch": per_epoch,
                "total_steps": total,
                "stop_step": stop,
                "world_size": world,
                "frozen_base_dependency": c["model"]["model_path"],
                "student_parameters": sum(p.numel() for p in params),
                "q_parameters": sum(p.numel() for p in q.parameters()),
                "update_order": "same pre-update Q for both losses, then one update each, then target EMA",
            },
        )
    start_time = time.monotonic()
    for step in range(start, stop):
        warm = c["warmup_steps"]
        mult = (
            min(1, (step + 1) / max(1, warm))
            * 0.5
            * (1 + math.cos(math.pi * max(0, step - warm) / max(1, total - warm)))
        )
        for group in opt.param_groups:
            group["lr"] = c["lr"] * mult
        opt.zero_grad(set_to_none=True)
        opt_q.zero_grad(set_to_none=True)
        metrics = {}
        local = data.indices(step, batch, c["seed"])[rank::world]
        tic = time.monotonic()
        for part in range(accum):
            b = data.batch(local[part * micro : (part + 1) * micro])
            tensor = {k: v.cuda(rank) for k, v in b.items() if torch.is_tensor(v)}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                next_context, next_features = prepare_fql(teacher.model, b["next_obs"])
                z = torch.randn(
                    (micro, 50, teacher.model.config.action_dim), device=rank
                )
                next_action = z - student(
                    z, torch.ones(micro, device=rank), next_context
                )
            with torch.no_grad():
                nq = target_q(
                    next_features,
                    next_action,
                    prefix_mask(tensor["remaining_after"], c["execute_chunk"]),
                    tensor["remaining_after"] / tensor["limit"],
                )
                target = bellman_target(
                    tensor["reward"], tensor["bootstrap_mask"], nq, c["discount"]
                )
            del next_context, next_features, next_action, nq, z
            with torch.autocast("cuda", dtype=torch.bfloat16):
                context, features = prepare_fql(teacher.model, b["obs"])
            predicted_q = q_run(
                features,
                tensor["action"],
                tensor["mask"],
                tensor["remaining_before"] / tensor["limit"],
            )
            q_loss = (predicted_q - target[:, None]).square().mean()
            if not torch.isfinite(q_loss):
                raise FloatingPointError("Nonfinite critic loss")
            (q_loss / accum).backward()
            # Freeze parameters only; retain the Q -> action derivative.
            q.requires_grad_(False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                noise = torch.randn(
                    (micro, 50, teacher.model.config.action_dim), device=rank
                )
                teacher_action = teacher.endpoint(
                    noise, context, steps=c["teacher_steps"]
                )
                prediction = noise - student_run(
                    noise, torch.ones(micro, device=rank), context
                )
            values = q(
                features,
                prediction,
                tensor["mask"],
                tensor["remaining_before"] / tensor["limit"],
            )
            actor_loss, parts = actor_objective(
                prediction, teacher_action, values, c["alpha"]
            )
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Nonfinite actor loss")
            # One output-gradient diagnostic, not a second full model backward.
            if step == start and part == 0:
                gm = torch.autograd.grad(
                    c["alpha"] * parts["endpoint_mse"], prediction, retain_graph=True
                )[0]
                gq = torch.autograd.grad(
                    parts["q_actor_loss"], prediction, retain_graph=True
                )[0]
                metrics.update(
                    anchor_action_grad_rms=float(gm.square().mean().sqrt()),
                    q_action_grad_rms=float(gq.square().mean().sqrt()),
                )
            (actor_loss / accum).backward()
            q.requires_grad_(True)
            row = dict(
                parts,
                actor_loss=actor_loss,
                critic_loss=q_loss,
                q_mean=predicted_q.mean(),
                q_abs_max=predicted_q.abs().max(),
                target_mean=target.mean(),
                reward_mean=tensor["reward"].mean(),
            )
            for k, v in row.items():
                metrics[k] = metrics.get(k, 0) + float(v.detach()) / accum
            del (
                context,
                features,
                prediction,
                teacher_action,
                values,
                actor_loss,
                parts,
                row,
                predicted_q,
                target,
                q_loss,
                b,
                tensor,
            )
        grad = torch.nn.utils.clip_grad_norm_(
            params, c["clip_grad"], error_if_nonfinite=True
        )
        # No extra critic clipping: official FQL uses its ordinary Adam update.
        q_norm = torch.nn.utils.clip_grad_norm_(
            q.parameters(), float("inf"), error_if_nonfinite=True
        )
        opt.step()
        opt_q.step()
        update_target(target_q, q, c["target_tau"])
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is None for p in target_q.parameters())
        if step == start or (step + 1) % c["log_every"] == 0 or step + 1 == stop:
            metrics.update(
                step=step + 1,
                epoch=(step + 1) / per_epoch,
                grad_norm=float(grad),
                q_grad_norm=float(q_norm),
                seconds_per_update=time.monotonic() - tic,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                lr=opt.param_groups[0]["lr"],
                time=time.time(),
                rank=rank,
            )
            with (out / f"metrics-rank{rank}.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            if rank == 0:
                save_json(out / "progress.json", metrics)
                print(json.dumps(metrics), flush=True)
        if not args.capacity_only and (
            (step + 1) % c["checkpoint_steps"] == 0 or step + 1 == stop
        ):
            rng = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(rank),
            }
            all_rng = [rng] if world == 1 else [None] * world
            if world > 1:
                dist.all_gather_object(all_rng, rng)
            if rank == 0:
                checkpoint = out / f"checkpoint-{step + 1:06d}.pt"
                policy = out / f"policy-{step + 1:06d}.pt"
                atomic_torch(
                    checkpoint,
                    {
                        "version": 1,
                        "config": c,
                        "step": step + 1,
                        "manifest": data.manifest,
                        "world_size": world,
                        "student": expert_state(student.model),
                        "q": q.state_dict(),
                        "target_q": target_q.state_dict(),
                        "optimizer": opt.state_dict(),
                        "q_optimizer": opt_q.state_dict(),
                        "rng": all_rng,
                    },
                )
                atomic_torch(
                    policy,
                    {
                        "expert": expert_state(student.model),
                        "method": "fql_fixed_teacher",
                        "step": step + 1,
                        "base": c["model"]["model_path"],
                    },
                )
                save_json(
                    out / "checkpoint.json",
                    {"step": step + 1, "path": str(checkpoint), "policy": str(policy)},
                )
                prune_checkpoints(
                    out,
                    c.get("checkpoint_keep_last", 2),
                    c.get("checkpoint_keep_steps", []),
                    single_task_smoke=bool(c.get("single_task")),
                )
            if world > 1:
                dist.barrier()
    if args.check_export and not args.capacity_only:
        # Run on the exact deployment factory/transform path after saving state.
        policy = teacher.model
        export = torch.load(
            out / f"policy-{stop:06d}.pt", map_location="cpu", weights_only=True
        )
        load_expert(policy, export["expert"])
        policy.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        policy.eval()
        calls = []
        original = policy.get_suffix_out

        def counted(*a, **kw):
            calls.append(1)
            return original(*a, **kw)

        policy.get_suffix_out = counted
        with torch.inference_mode():
            actions, _ = policy.predict_action_batch(
                dict(sample_obs, extra_view_images=None),
                mode="eval",
                compute_values=False,
                num_steps=1,
            )
        assert (
            len(calls) == 1
            and tuple(actions.shape) == (1, 50, 14)
            and torch.isfinite(torch.as_tensor(actions)).all()
        )
        save_json(
            out / f"export-check-rank{rank}.json",
            {"calls": 1, "shape": list(actions.shape), "finite": True, "step": stop},
        )
    if rank == 0:
        save_json(
            out / "segment-done.json",
            {"step": stop, "elapsed_seconds": time.monotonic() - start_time},
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
