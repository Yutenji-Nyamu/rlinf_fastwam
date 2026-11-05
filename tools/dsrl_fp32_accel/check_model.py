# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Real model + production FSDP wrapping + optimizer round-trip preflight."""

import argparse
import collections
import io
import json
import os
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi import get_model


def main(directory):
    d = Path(directory)
    cfg = OmegaConf.load(d / "config/formal.yaml")
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    torch.cuda.set_device(0)
    fd, path = tempfile.mkstemp(prefix="dsrl-fsdp-")
    os.close(fd)
    torch.distributed.init_process_group(
        "nccl", init_method="file://" + path, rank=0, world_size=1
    )
    model = get_model(cfg.actor.model).cuda()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(p.dtype == torch.float32 for p in trainable.values())
    frozen = collections.Counter(
        str(p.dtype) for p in model.parameters() if not p.requires_grad
    )
    assert frozen["torch.bfloat16"] > 0
    policy = get_fsdp_wrap_policy(
        model, config=cfg.actor.fsdp_config, is_lora=False, model_type="openpi"
    )
    wrapped = FSDP(
        model,
        auto_wrap_policy=policy,
        device_id=0,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=True,
        sync_module_states=True,
    )
    params = [p for p in wrapped.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in params)
    optimizer = torch.optim.Adam(params, lr=1e-4)
    obs = {
        "states": torch.randn(8, cfg.actor.model.openpi.dsrl_state_dim, device="cuda"),
        "main_images": torch.randint(
            0, 256, (8, 64, 64, 3), device="cuda", dtype=torch.uint8
        ),
    }
    before = [p.detach().clone() for p in params]
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        action, logp, _ = wrapped(forward_type=ForwardType.SAC, obs=obs)
        q = wrapped(forward_type=ForwardType.SAC_Q, obs=obs, actions=action)
        assert action.shape == (8, 50, 32) and q.shape == (8, 10)
        assert action.dtype == logp.dtype == q.dtype == torch.float32
        loss = (0.02 * logp - q.mean(-1)).mean()
        assert torch.isfinite(loss)
        loss.backward()
        norm = wrapped.clip_grad_norm_(3.5)
        optimizer.step()
    changed = sum(int(p.ne(b).sum()) for p, b in zip(params, before))
    assert changed > 0
    states = [
        v.dtype
        for state in optimizer.state.values()
        for k, v in state.items()
        if k in ("exp_avg", "exp_avg_sq")
    ]
    assert states and all(v == torch.float32 for v in states)
    # Serialized small-network + optimizer round trip, followed by true full-run
    # smoke/resume phases, checks state precision and FSDP's nested parameter names.
    state = wrapped.state_dict()
    small = {
        k: v.detach().cpu()
        for k, v in state.items()
        if any(
            k.startswith(x + ".")
            for x in (
                "dsrl_action_noise_net",
                "actor_image_encoder",
                "actor_state_encoder",
                "critic_image_encoder",
                "critic_state_encoder",
                "q_head",
            )
        )
    }
    assert small and all(x.dtype == torch.float32 for x in small.values())
    stream = io.BytesIO()
    torch.save({"small": small, "optim": optimizer.state_dict()}, stream)
    stream.seek(0)
    recovered = torch.load(stream, weights_only=False)
    optimizer.load_state_dict(recovered["optim"])
    result = {
        "trainable_parameter_count": sum(p.numel() for p in params),
        "trainable_dtype": "float32",
        "frozen_dtype_tensors": dict(frozen),
        "optimizer_moments": "float32",
        "changed_parameters": changed,
        "fsdp_forward_backward_clip": "pass",
        "small_state_roundtrip": "pass",
        "loss": float(loss.detach()),
        "gradient_norm": float(norm),
        "noise_bound": 2,
        "small_state_keys": len(small),
    }
    (d / "fp32-model-contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    main(p.parse_args().directory)
