# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Frozen-prefix caching for the accepted RLT token-only reconstruction loss.

The existing token-only sft_forward extracts its frozen VLA prefix with
train=False. Cache those exact tensors once; keep the same causal decoder,
mask, FP32 token training, optimizer and 32-example optimizer batches.
"""

import gc
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def dataset_spec(plan):
    root = Path(plan["stage1_data"])
    info = json.loads((root / "meta/info.json").read_text())
    tasks = {
        x["task_index"]: x["task"]
        for x in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())
    }
    subset = root / "subset-manifest.json"
    if subset.exists():
        m = json.loads(subset.read_text())
        train = m["train"]
        val = m["validation"]
        source = hashlib.sha256(subset.read_bytes()).hexdigest()
    else:
        train = list(
            map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())
        )
        val = []
        source = hashlib.sha256((root / "meta/episodes.jsonl").read_bytes()).hexdigest()
    return root, info, tasks, train, val, source


def rows(root, info, tasks, episodes, limit=None):
    import av
    import pyarrow.parquet as pq
    from PIL import Image

    total = sum(x["length"] for x in episodes)
    chosen = (
        None
        if limit is None
        else set(np.linspace(0, total - 1, min(limit, total), dtype=int).tolist())
    )
    global_index = 0
    cameras = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    for episode in episodes:
        kw = {
            "episode_index": episode["episode_index"],
            "episode_chunk": episode["episode_index"] // info["chunks_size"],
        }
        table = pq.read_table(root / info["data_path"].format(**kw)).to_pydict()
        assert len(table["timestamp"]) == episode["length"]
        containers = []
        streams = []
        if info["features"][cameras[0]]["dtype"] == "video":
            for key in cameras:
                c = av.open(str(root / info["video_path"].format(video_key=key, **kw)))
                containers.append(c)
                streams.append(iter(c.decode(video=0)))
        try:
            for i in range(episode["length"]):
                decoded = [next(s) for s in streams]
                take = chosen is None or global_index in chosen
                global_index += 1
                if not take:
                    continue
                images = []
                for k, key in enumerate(cameras):
                    if streams:
                        frame = decoded[k]
                        assert (
                            abs(float(frame.time) - float(table["timestamp"][i]))
                            <= 1.01 / info["fps"]
                        )
                        images.append(frame.to_ndarray(format="rgb24"))
                    else:
                        v = table[key][i]
                        raw = v["bytes"] if isinstance(v, dict) else v
                        if raw is None:
                            raw = (root / v["path"]).read_bytes()
                        images.append(
                            np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                        )
                yield {
                    "images": images,
                    "state": np.asarray(
                        table["observation.state"][i], dtype=np.float32
                    ),
                    "prompt": tasks[int(table["task_index"][i])],
                }
        finally:
            for c in containers:
                c.close()


def observation(model, batch):
    from openpi.models import model as model_api

    raw = {
        "observation/image": np.stack([x["images"][0] for x in batch]),
        "observation/wrist_image": np.stack([np.stack(x["images"][1:]) for x in batch]),
        "observation/state": np.stack([x["state"] for x in batch]),
        "prompt": [x["prompt"] for x in batch],
    }
    processed = model.precision_processor(model.input_transform(raw, transpose=False))
    return model_api.Observation.from_dict(processed)


def cache(model, folder, name, iterator, count, contract):
    meta = folder / (name + ".json")
    if meta.exists():
        m = json.loads(meta.read_text())
        assert m["contract"] == contract and m["count"] == count
        return m
    prefix_map = mask_map = None
    offset = 0
    batch = []
    start = time.time()

    def flush(batch):
        nonlocal prefix_map, mask_map, offset
        with torch.no_grad():
            prefix, mask = model._extract_rlt_prefix_embeddings(
                observation(model, batch), train=False
            )
        prefix = prefix.detach().cpu().contiguous()
        mask = mask.cpu().bool()
        assert prefix.dtype == torch.bfloat16, (
            f"Unexpected source dtype {prefix.dtype}; refuse quantizing cached features"
        )
        if prefix_map is None:
            assert tuple(prefix.shape[1:]) == (768, 2048)
            prefix_map = np.memmap(
                folder / (name + ".bf16"),
                dtype=np.uint16,
                mode="w+",
                shape=(count, 768, 2048),
            )
            mask_map = np.memmap(
                folder / (name + ".mask"), dtype=np.bool_, mode="w+", shape=(count, 768)
            )
            assert torch.isfinite(prefix).all()
        n = len(batch)
        prefix_map[offset : offset + n] = prefix.view(torch.uint16).numpy()
        mask_map[offset : offset + n] = mask.numpy()
        # Verify the cache representation is bitwise identical to the source.
        if offset == 0:
            assert torch.equal(
                torch.from_numpy(prefix_map[:n].copy()).view(torch.bfloat16), prefix
            )
        offset += n
        if offset % 512 == 0:
            print(
                json.dumps(
                    {
                        "phase": "prefix_cache",
                        "split": name,
                        "frames": offset,
                        "total": count,
                        "seconds": time.time() - start,
                    }
                ),
                flush=True,
            )

    for row in iterator:
        batch.append(row)
        if len(batch) == 8:
            flush(batch)
            batch = []
    if batch:
        flush(batch)
    assert offset == count
    prefix_map.flush()
    mask_map.flush()
    m = {
        "contract": contract,
        "count": count,
        "shape": [count, 768, 2048],
        "dtype": "bfloat16",
        "seconds": time.time() - start,
    }
    save_json(meta, m)
    return m


def main(directory):
    d = Path(directory)
    plan = json.loads((d / "plan.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(plan["gpu"])
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.load(d / "config/stage1.yaml")
    root, info, tasks, train, val, source = dataset_spec(plan)
    folder = Path(plan["run"]) / "stage1"
    folder.mkdir(parents=True, exist_ok=True)
    contract = {
        "base_model": str(cfg.model_path),
        "config_name": cfg.openpi.config_name,
        "prefix_seq_len": 768,
        "embed_dim": 2048,
    }
    cache_contract = {
        **contract,
        "dataset": source,
        "feature_train": False,
        "source": plan["rlt_source_sha"],
    }
    model = get_model(cfg).cuda().eval()
    assert all(
        n.startswith("rlt_module.")
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    cache(
        model,
        folder,
        "train",
        rows(root, info, tasks, train),
        sum(x["length"] for x in train),
        cache_contract,
    )
    if val:
        cache(
            model,
            folder,
            "validation",
            rows(root, info, tasks, val, limit=256),
            min(256, sum(x["length"] for x in val)),
            cache_contract,
        )
    module = model.rlt_module
    del model
    gc.collect()
    torch.cuda.empty_cache()
    module.train()
    count = sum(x["length"] for x in train)
    data = np.memmap(
        folder / "train.bf16", dtype=np.uint16, mode="r", shape=(count, 768, 2048)
    )
    masks = np.memmap(
        folder / "train.mask", dtype=np.bool_, mode="r", shape=(count, 768)
    )

    def batch(indices):
        x = (
            torch.from_numpy(np.array(data[indices], copy=True))
            .view(torch.bfloat16)
            .cuda()
            .float()
        )
        m = torch.from_numpy(np.array(masks[indices], copy=True)).cuda()
        return x, m

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        fused=True,
    )

    def lr_factor(step):
        if step < 100:
            return (step + 1) / 100
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (step - 100) / 1900)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    rng = np.random.default_rng(0)
    start_step = 0
    checkpoint = folder / "resume.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["contract"] == contract
        module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start_step = state["step"]
    initial = None
    start = time.time()
    for step in range(start_step, 2000):
        optimizer.zero_grad(set_to_none=True)
        indices = rng.integers(0, count, size=32)
        losses = []
        for ids in np.split(indices, 4):
            x, m = batch(ids)
            loss, _ = module(x, m)
            (loss / 4).backward()
            losses.append(loss.detach())
        loss = torch.stack(losses).mean()
        assert torch.isfinite(loss)
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        scheduler.step()
        if initial is None:
            initial = float(loss)
        if (step + 1) % 50 == 0 or step == start_step:
            record = {
                "phase": "token_train",
                "step": step + 1,
                "loss": float(loss),
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - start,
            }
            print(json.dumps(record), flush=True)
            with (folder / "training.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        if (step + 1) % 500 == 0:
            payload = {
                "contract": contract,
                "model": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step + 1,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            }
            tmp = folder / "resume.tmp"
            torch.save(payload, tmp)
            tmp.replace(checkpoint)
    module.eval()
    with torch.no_grad():
        if val:
            n = min(256, sum(x["length"] for x in val))
            v = np.memmap(
                folder / "validation.bf16",
                dtype=np.uint16,
                mode="r",
                shape=(n, 768, 2048),
            )
            vm = np.memmap(
                folder / "validation.mask", dtype=np.bool_, mode="r", shape=(n, 768)
            )
            vl = []
            for offset in range(0, n, 8):
                x = (
                    torch.from_numpy(np.array(v[offset : offset + 8], copy=True))
                    .view(torch.bfloat16)
                    .cuda()
                    .float()
                )
                m = torch.from_numpy(
                    np.array(vm[offset : offset + 8], copy=True)
                ).cuda()
                vl.append(module(x, m)[0].detach())
            validation = float(torch.stack(vl).mean())
        else:
            validation = None
    state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    target = folder / "token-final.pt"
    tmp = folder / "token-final.tmp"
    torch.save({"contract": contract, "model": state, "step": 2000}, tmp)
    tmp.replace(target)
    receipt = {
        "contract": contract,
        "step": 2000,
        "train_frames": count,
        "initial_train_loss": initial,
        "last_train_loss": float(loss),
        "validation_loss": validation,
        "token_path": str(target),
        "token_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "cache_contract": cache_contract,
    }
    save_json(folder / "complete.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
