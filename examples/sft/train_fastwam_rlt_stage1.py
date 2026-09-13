"""Freeze Fast-WAM features; train the existing full causal-AR RLT module.

Run extract once, then train. Cached tensors contain final current-frame hidden,
not layer KV or action targets. --max-frames/--steps are explicit smoke budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.fastwam.builder import build_fastwam_policy
from rlinf.models.embodiment.fastwam.rlt_features import build_rlt_conditioning
from rlinf.models.embodiment.fastwam.rlt_policy import feature_identity
from rlinf.models.embodiment.fastwam.rlt_stage1_data import lerobot_frame_to_env_obs
from rlinf.models.embodiment.fastwam.robotwin_adapter import adapt_robotwin_observation
from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenTransformer


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def extract(cfg, out, max_frames):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    policy = build_fastwam_policy(cfg.model, torch_dtype=torch.bfloat16)
    policy.requires_grad_(False).eval()
    dataset = LeRobotDataset(
        repo_id=str(cfg.data.repo_id), root=Path(cfg.data.path),
        download_videos=False,
    )
    count = len(dataset) if max_frames <= 0 else min(max_frames, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, count).round().long().tolist()
    token_kwargs = OmegaConf.to_container(cfg.token, resolve=True)
    identity = feature_identity(cfg.model.checkpoint_path, cfg.model.dataset_stats_path, token_kwargs)
    shards, pending, first_obs = [], [], None
    for position, index in enumerate(indices):
        obs = lerobot_frame_to_env_obs(dataset[index], str(cfg.data.default_prompt))
        if first_obs is None:
            first_obs = obs
            torch.save(first_obs, out / "first-observation.pt")
        with torch.no_grad():
            image, proprio, prompts = adapt_robotwin_observation(
                obs, policy.processor, device=policy.device, dtype=policy.model_dtype,
            )
            text, mask = policy.model.encode_prompt(prompts)
            _, hidden = build_rlt_conditioning(
                policy.model, input_image=image, text_context=text,
                text_context_mask=mask, proprio=proprio, retain_cache=False,
            )
        if tuple(hidden.shape[1:]) != (token_kwargs["prefix_seq_len"], token_kwargs["input_dim"]):
            raise ValueError(f"Expected configured Fast-WAM hidden shape, got {hidden.shape}")
        pending.append(hidden.cpu().to(torch.bfloat16))
        if len(pending) >= 128 or position == count - 1:
            name = f"features-{len(shards):04d}.pt"
            tensor = torch.cat(pending)
            torch.save(tensor, out / name)
            shards.append({"path": name, "count": tensor.shape[0]})
            pending.clear()
            print(json.dumps({"extracted": position + 1, "total": count}), flush=True)
    manifest = {
        "identity": identity, "dataset_path": str(cfg.data.path),
        "dataset_total_frames": len(dataset), "selected_indices": indices,
        "frames": count, "shards": shards,
        "smoke_subset": count != len(dataset),
    }
    write_json(out / "features.json", manifest)


def train(cfg, out, feature_dir, steps, resume):
    manifest = json.loads((feature_dir / "features.json").read_text())
    identity = manifest["identity"]
    expected = feature_identity(
        cfg.model.checkpoint_path, cfg.model.dataset_stats_path,
        OmegaConf.to_container(cfg.token, resolve=True),
    )
    if identity != expected:
        raise ValueError("Cached Fast-WAM features do not match teacher/config")
    torch.manual_seed(int(cfg.seed))
    # At 7188 x 120 x 3072 BF16 the complete clean50 cache is about 4.94 GiB.
    features = torch.cat([
        torch.load(feature_dir / item["path"], map_location="cpu", weights_only=True)
        for item in manifest["shards"]
    ])
    generator = torch.Generator().manual_seed(int(cfg.seed))
    model = RLTTokenTransformer(**identity["token_kwargs"]).cuda().float()
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.optim.lr),
        betas=(float(cfg.optim.beta1), float(cfg.optim.beta2)),
        eps=float(cfg.optim.eps), weight_decay=float(cfg.optim.weight_decay),
    )
    initial_step = 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Stage1 resume identity mismatch")
        model.load_state_dict(saved["token_transformer"])
        opt.load_state_dict(saved["optimizer"])
        initial_step = saved["step"]
        generator.set_state(saved["sample_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    total_steps = int(cfg.steps)
    target_step = total_steps if steps is None else int(steps)
    batch_size, micro = int(cfg.batch_size), int(cfg.micro_batch_size)
    if batch_size % micro:
        raise ValueError("Stage1 batch_size must be divisible by micro_batch_size")
    model.train()
    logs = out / "metrics.jsonl"
    for step in range(initial_step, target_step):
        # Uniform without replacement within a full batch. Tiny smoke subsets
        # repeat only after every distinct cached frame has been selected.
        selected = []
        while len(selected) < batch_size:
            selected.extend(torch.randperm(len(features), generator=generator).tolist())
        selected = selected[:batch_size]
        warmup = int(cfg.optim.warmup_steps)
        if step < warmup:
            rate = (step + 1) / max(warmup, 1)
        else:
            progress = min(1., (step - warmup) / max(total_steps - warmup, 1))
            rate = float(cfg.optim.min_lr_rate) + (1 - float(cfg.optim.min_lr_rate)) * .5 * (1 + math.cos(math.pi * progress))
        for group in opt.param_groups:
            group["lr"] = float(cfg.optim.lr) * rate
        opt.zero_grad(set_to_none=True)
        mse = 0.
        for start in range(0, batch_size, micro):
            hidden = features[selected[start:start + micro]].cuda().float()
            loss, _ = model.loss(hidden)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Stage1 loss")
            (loss * micro / batch_size).backward()
            mse += float(loss.detach()) * micro / batch_size
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.clip_grad))
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite Stage1 gradient")
        opt.step()
        row = {"step": step + 1, "mse": mse, "grad_norm": float(norm), "lr": opt.param_groups[0]["lr"]}
        with logs.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    checkpoint = {
        "identity": identity, "step": target_step,
        "token_transformer": model.cpu().state_dict(),
        "encoder": model.encoder.state_dict(), "optimizer": opt.state_dict(),
        "sample_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "smoke": bool(manifest["smoke_subset"] or target_step < total_steps),
    }
    checkpoint_path = out / "stage1.pt"
    torch.save(checkpoint, checkpoint_path)
    stage1_manifest = {
        "identity": identity, "step": target_step, "smoke": checkpoint["smoke"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "feature_manifest_sha256": hashlib.sha256((feature_dir / "features.json").read_bytes()).hexdigest(),
    }
    write_json(out / "stage1-manifest.json", stage1_manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["extract", "train"])
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--features")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--steps", type=int)
    p.add_argument("--resume")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "extract":
        extract(cfg, out, args.max_frames)
    else:
        if not args.features:
            p.error("train requires --features")
        train(cfg, out, Path(args.features), args.steps, args.resume)


if __name__ == "__main__":
    main()
