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

"""Two-GPU, offline pi05 distillation with full expert/optimizer/RNG resume state."""

import argparse
import copy
import json
import math
import os
import random
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from rlinf.algorithms.pi05_distillation.checkpoint_retention import prune_checkpoints
from rlinf.algorithms.pi05_distillation.data import BalancedWindows, DemoWindows
from rlinf.algorithms.pi05_distillation.losses import (
    cp_loss,
    ema_decay,
    endpoint_loss,
    fake_loss,
    onedp_loss,
)
from rlinf.algorithms.pi05_distillation.model import (
    Velocity,
    expert_state,
    load_expert,
    prepare,
)
from rlinf.algorithms.pi05_distillation.prefetch import decoded_batches, training_indices
from rlinf.models.embodiment.openpi import get_model


def save_json(path, value):
    """Atomically publish a small, readable progress receipt."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def atomic_torch(path, value):
    """Finish a checkpoint before publishing its path to the queue."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(value, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--decode-workers", type=int, default=0)
    parser.add_argument("--prefetch-batches", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ.get(
        "DISTILL_GPU_PAIR", "2,3"
    )
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cuda.enable_cudnn_sdp(False)
    dist.init_process_group("nccl")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(c["seed"] + rank)
    np.random.seed(c["seed"] + rank)
    torch.manual_seed(c["seed"] + rank)
    data = (
        BalancedWindows(c["data_manifest"])
        if c.get("data_manifest")
        else DemoWindows(c["data"])
    )
    global_batch, micro = c["batch_size"], c["microbatch_per_rank"]
    assert global_batch % (world * micro) == 0
    accum = global_batch // (world * micro)
    per_epoch = math.ceil(len(data) / global_batch)
    total = c["epochs"] * per_epoch
    stop = min(args.stop_step or total, total)
    method = c["method"]
    if method not in {"cp", "onedp", "endpoint"}:
        raise ValueError(f"Unknown distillation method: {method}")
    cfg = copy.deepcopy(c["model"])
    cfg["openpi"].update(
        distill_target_time=method == "cp",
        noise_method="flow_ode",
        use_rlt=False,
        use_dsrl=False,
        is_nft=False,
        rtc_enabled=False,
    )
    teacher_model = get_model(OmegaConf.create(cfg))
    student_model = copy.deepcopy(teacher_model)
    aux_model = copy.deepcopy(student_model) if method != "endpoint" else None
    # All policies have an identical frozen VLM. Share those weights and
    # compute its observation prefix once, as the accepted native expert path.
    for model in (student_model, aux_model):
        if model is None:
            continue
        model.paligemma_with_expert.paligemma = (
            teacher_model.paligemma_with_expert.paligemma
        )
        # The unused vocabulary projection is part of GemmaForCausalLM, not the
        # action expert. Keep it frozen/shared; it has no path to action loss.
        model.paligemma_with_expert.gemma_expert.lm_head = (
            teacher_model.paligemma_with_expert.gemma_expert.lm_head
        )
    teacher = Velocity(teacher_model, False).cuda(rank)
    student = Velocity(student_model, True).cuda(rank)
    auxiliary = (
        Velocity(aux_model, method == "onedp").cuda(rank)
        if aux_model is not None
        else None
    )
    student_params = student.parameters_to_train()
    opt = torch.optim.AdamW(
        student_params,
        lr=c["lr"],
        betas=tuple(c["betas"]),
        eps=1e-8,
        weight_decay=c["weight_decay"],
    )
    opt_aux = (
        torch.optim.Adam(
            auxiliary.parameters_to_train(),
            lr=c["aux_lr"],
            betas=(0.0, 0.999),
            eps=1e-8,
        )
        if method == "onedp"
        else None
    )
    start = 0
    if args.resume:
        # Trusted local training state contains Python/NumPy RNG, not public data.
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck["config"] != c or ck["windows"] != len(data) or ck["world_size"] != world:
            raise ValueError("Resume configuration/data/world-size mismatch")
        if (
            isinstance(data, BalancedWindows)
            and ck.get("task_manifest") != data.manifest
        ):
            raise ValueError("Resume task manifest/window counts changed")
        load_expert(student.model, ck["student"])
        if auxiliary is not None:
            load_expert(auxiliary.model, ck["auxiliary"])
        elif ck["auxiliary"] is not None:
            raise ValueError(
                "Endpoint checkpoint must not contain an auxiliary network"
            )
        opt.load_state_dict(ck["optimizer"])
        if opt_aux:
            opt_aux.load_state_dict(ck["aux_optimizer"])
        start = ck["step"]
        rng = ck["rng"][rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], rank)
        del ck
    student_ddp = DDP(student, device_ids=[rank], broadcast_buffers=False)
    aux_ddp = (
        DDP(auxiliary, device_ids=[rank], broadcast_buffers=False)
        if opt_aux
        else auxiliary
    )
    if rank == 0:
        save_json(
            out / "resolved.json",
            {
                "config": c,
                "windows": len(data),
                "steps_per_epoch": per_epoch,
                "total_steps": total,
                "stop_step": stop,
                "resume": args.resume,
                "world_size": world,
                "frozen_base_dependency": c["model"]["model_path"],
                "mutable_parameters": sum(p.numel() for p in student_params),
                "task_manifest": getattr(data, "manifest", None),
                "cpu_loader": {
                    "decode_workers": args.decode_workers,
                    "prefetch_batches": args.prefetch_batches,
                },
            },
        )
    begin = time.monotonic()
    batches = decoded_batches(
        data,
        training_indices(data, start, stop, global_batch, micro, rank, world, c["seed"]),
        decode_workers=args.decode_workers,
        prefetch_batches=args.prefetch_batches,
    )
    with closing(batches):
        for step in range(start, stop):
            lr_mult = 1.0
            if method in {"cp", "endpoint"}:
                warm = c["warmup_steps"]
                lr_mult = (
                    min(1.0, (step + 1) / max(1, warm))
                    * 0.5
                    * (1 + math.cos(math.pi * max(0, step - warm) / max(1, total - warm)))
                )
            for group in opt.param_groups:
                group["lr"] = c["lr"] * lr_mult
            opt.zero_grad(set_to_none=True)
            if opt_aux:
                opt_aux.zero_grad(set_to_none=True)
            metrics = {}
            # Cache each microbatch's frozen context for fake and generator updates.
            prepared = []
            for m in range(accum):
                obs, actions = next(batches)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actions, context = prepare(teacher.model, obs, actions)
                prepared.append((actions, context))
            if opt_aux:
                for actions, context in prepared:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = torch.randn_like(actions)
                        with torch.no_grad():
                            generated = z - student(
                                z, torch.ones(len(z), device=z.device), context
                            )
                        lf = fake_loss(aux_ddp, generated, context)
                    (lf / accum).backward()
                    metrics["fake"] = metrics.get("fake", 0.0) + lf.detach() / accum
                aux_norm = torch.nn.utils.clip_grad_norm_(
                    auxiliary.parameters_to_train(), c["clip_grad"], error_if_nonfinite=True
                )
                opt_aux.step()
                opt_aux.zero_grad(set_to_none=True)
                metrics["fake_grad_norm"] = aux_norm
            for actions, context in prepared:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if method == "cp":
                        loss, parts = cp_loss(
                            student_ddp,
                            teacher,
                            auxiliary,
                            actions,
                            context,
                            bins=c["time_bins"],
                        )
                    elif method == "endpoint":
                        z = torch.randn_like(actions)
                        if step == 0 and out.name.startswith("smoke-"):
                            # Native N1 and the differentiable student expression
                            # must agree before training. Only one smoke batch.
                            with torch.no_grad():
                                native = teacher.endpoint(z, context, steps=1)
                                direct = z - student(
                                    z, torch.ones(len(z), device=z.device), context
                                )
                                torch.testing.assert_close(
                                    direct, native, rtol=1e-5, atol=1e-6
                                )
                            save_json(
                                out / f"native-parity-rank{rank}.json",
                                {
                                    "max_abs": float((direct - native).abs().max()),
                                    "teacher_target": "native_eval_flow_ode_N10",
                                    "student_initial_N1_matches": True,
                                },
                            )
                            del native, direct
                        loss, parts = endpoint_loss(
                            student_ddp,
                            teacher,
                            z,
                            context,
                            teacher_steps=c["teacher_steps"],
                            action_dim=c["loss_action_dim"],
                        )
                    elif method == "onedp":
                        z = torch.randn_like(actions)
                        generated = z - student_ddp(
                            z, torch.ones(len(z), device=z.device), context
                        )
                        loss, score = onedp_loss(generated, teacher, auxiliary, context)
                        parts = {"score_gradient_rms": score}
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite distillation loss")
                (loss / accum).backward()
                for k, v in dict(parts, loss=loss.detach()).items():
                    metrics[k] = metrics.get(k, 0.0) + v.detach() / accum
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_params, c["clip_grad"], error_if_nonfinite=True
            )
            opt.step()
            if method == "cp":
                decay = ema_decay(step + 1)
                with torch.no_grad():
                    src = dict(student.model.named_parameters())
                    for name, p in auxiliary.model.named_parameters():
                        if src[name].requires_grad:
                            p.lerp_(src[name], 1 - decay)
            del prepared, actions, context, loss
            if (step + 1) % c["log_every"] == 0 or step == start or step + 1 == stop:
                row = {k: float(v) for k, v in metrics.items()}
                row.update(
                    step=step + 1,
                    epoch=(step + 1) / per_epoch,
                    grad_norm=float(grad_norm),
                    lr=opt.param_groups[0]["lr"],
                    seconds_per_update=(time.monotonic() - begin) / (step + 1 - start),
                    peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    time=time.time(),
                    rank=rank,
                )
                with (out / f"metrics-rank{rank}.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    save_json(out / "progress.json", row)
            if (step + 1) % c["checkpoint_steps"] == 0 or step + 1 == stop:
                rng = {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(rank),
                }
                all_rng = [None] * world
                dist.all_gather_object(all_rng, rng)
                if rank == 0:
                    checkpoint = out / f"checkpoint-{step + 1:06d}.pt"
                    atomic_torch(
                        checkpoint,
                        {
                            "version": 1,
                            "config": c,
                            "step": step + 1,
                            "windows": len(data),
                            "world_size": world,
                            "student": expert_state(student.model),
                            "auxiliary": expert_state(auxiliary.model)
                            if auxiliary is not None
                            else None,
                            "optimizer": opt.state_dict(),
                            "aux_optimizer": opt_aux.state_dict() if opt_aux else None,
                            "rng": all_rng,
                            "task_manifest": getattr(data, "manifest", None),
                            "task_exposure": (
                                data.exposure((step + 1) * global_batch, c["seed"])
                                if isinstance(data, BalancedWindows)
                                else None
                            ),
                        },
                    )
                    # CP uses its EMA policy at evaluation, following the reference.
                    policy_path = out / f"policy-{step + 1:06d}.pt"
                    atomic_torch(
                        policy_path,
                        {
                            "expert": expert_state(
                                auxiliary.model if method == "cp" else student.model
                            ),
                            "method": method,
                            "step": step + 1,
                            "base": c["model"]["model_path"],
                        },
                    )
                    save_json(
                        out / "checkpoint.json",
                        {
                            "step": step + 1,
                            "path": str(checkpoint),
                            "policy": str(policy_path),
                        },
                    )
                    prune_checkpoints(
                        out, c.get("checkpoint_keep_last"), c.get("checkpoint_keep_steps", [])
                    )
                dist.barrier()
    if rank == 0:
        save_json(
            out / "segment-done.json",
            {
                "step": stop,
                "total_steps": total,
                "complete": stop == total,
                "elapsed_seconds": time.monotonic() - begin,
            },
        )
    if out.name.startswith("smoke-") and stop == 14:
        assert all(p.grad is None for p in teacher.parameters())
        # Exercise the actual exported overlay in the factory's inference dtype,
        # after the resumable training state has already been saved.
        export = torch.load(
            out / f"policy-{stop:06d}.pt", map_location="cpu", weights_only=True
        )
        policy = teacher.model
        load_expert(policy, export["expert"])
        policy.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        del export
        # Multitask training capacity is independent of simulator parallelism.
        # Exercise the actual multitask evaluation batch, 16 per task/lane.
        inference_batch = (
            16
            if method == "endpoint"
            or (method == "onedp" and isinstance(data, BalancedWindows))
            else micro
        )
        obs, _ = data.batch(list(range(inference_batch)))
        obs["extra_view_images"] = None
        calls = []
        original = policy.get_suffix_out

        def counted(*positional, **kwargs):
            calls.append(1)
            return original(*positional, **kwargs)

        policy.get_suffix_out = counted
        with torch.inference_mode():
            action, _ = policy.predict_action_batch(
                obs, mode="eval", compute_values=False, num_steps=1
            )
        policy.get_suffix_out = original
        assert len(calls) == 1 and tuple(action.shape) == (inference_batch, 50, 14)
        assert torch.isfinite(torch.as_tensor(action)).all()
        assert all(p.grad is None for p in teacher.parameters())
        save_json(
            out / f"inference-check-rank{rank}.json",
            {
                "action_expert_calls": 1,
                "shape": list(action.shape),
                "finite": True,
                "teacher_frozen_during_training": True,
                "resumed_step": 14,
                "loaded_export": True,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "local_cuda_device": torch.cuda.current_device(),
            },
        )
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
