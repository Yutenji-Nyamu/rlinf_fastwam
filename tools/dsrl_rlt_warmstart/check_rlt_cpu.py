# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Check the port's dimensions, clean route, configuration, and EMA math."""

import json
import os
import sys
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.scheduler import Cluster
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

torch.set_num_threads(4)
Cluster.NAMESPACE = "RLinf_rlt_cpu_" + str(os.getpid())
assert callable(RLTACFSDPPolicy.forward_actor) and callable(
    EmbodiedSACFSDPPolicy.update_one_epoch
)
assert SupportedEnvType("robodojo") == SupportedEnvType.ROBODOJO
assert get_env_cls("robodojo").__name__ == "RoboDojoEnv"
B = Path("/data/chenyiteng/deployment-20261106/dsrl-rlt-warmstart")
results = []
for bench, gpu in [("dojo", 6), ("rt2", 7)]:
    if bench != sys.argv[1]:
        continue
    d = B / f"{bench}-rlt-clean-g{gpu}"
    c = validate_cfg(OmegaConf.load(d / "config/formal.yaml"))
    assert c.rollout.model.model_path is None
    assert (
        c.algorithm.rlt_dvac.mode == "off"
        and c.rollout.model.output_activation == "identity"
    )
    assert (
        c.actor.model.num_action_chunks
        == c.rollout.rlt_feature_model.openpi.action_chunk
    )
    f = c.rollout.rlt_feature_model
    dc = get_openpi_config(
        f.openpi.config_name, model_path=f.model_path, data_kwargs=f.openpi_data
    )
    data = dc.data.create(dc.assets_dirs, dc.model)
    model = RLTMLPPolicy(
        z_dim=2048,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=c.actor.model.num_action_chunks,
        output_activation="identity",
    )
    obs = {
        "z_rl": torch.randn(3, 2048),
        "proprio": torch.randn(3, 14),
        "ref_chunk": torch.randn(3, c.actor.model.num_action_chunks, 14),
    }
    action, _, _ = model.sac_forward(obs)
    assert action.shape == (
        3,
        c.actor.model.num_action_chunks * 14,
    ) or action.shape == (3, c.actor.model.num_action_chunks, 14)
    assert torch.isfinite(action).all()
    action.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    results.append(
        {
            "bench": bench,
            "chunk": c.actor.model.num_action_chunks,
            "action_shape": list(action.shape),
            "quantile_norm": data.use_quantile_norm,
            "dtype": str(next(model.parameters()).dtype),
        }
    )
# Same multiplication order as the scalar implementation, batched across tensors.
a = [torch.randn(17), torch.randn(13)]
b = [torch.randn(17), torch.randn(13)]
expected = [x * 0.995 + y * 0.005 for x, y in zip(a, b)]
torch._foreach_mul_(a, 0.995)
torch._foreach_add_(a, torch._foreach_mul(b, 0.005))
assert all(torch.equal(x, y) for x, y in zip(a, expected))
print(
    json.dumps(
        {
            "port_configs": results,
            "foreach_ema_bitwise_cpu": True,
            "worker_imports": True,
        }
    )
)
ray.shutdown()
