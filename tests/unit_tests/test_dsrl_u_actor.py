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

"""CPU checks for the opt-in actor allocation and strict trainer contract."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.dsrl_ugrow import U_SPEC, build_dsrl_u_weights
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class _SmallActorQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.2))

    def forward(self, *, forward_type, obs, actions=None, **kwargs):
        if forward_type == ForwardType.SAC:
            action = self.theta * obs["states"]
            return action, 2 * action, None
        if forward_type == ForwardType.SAC_Q:
            return (3 * actions).expand(-1, 2)
        raise AssertionError(forward_type)


def _worker(enabled=True):
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "global_batch_size": 4,
                "model": {
                    "model_type": "openpi",
                    "num_q_heads": 2,
                    "num_action_chunks": 10,
                    "openpi": {
                        "use_dsrl": True,
                        "dsrl_u_enabled": enabled,
                        "dsrl_u_spec": copy.deepcopy(U_SPEC),
                    },
                },
            },
            "algorithm": {
                "agg_q": "mean",
                "actor_agg_q": "mean",
                "bootstrap_type": "standard",
                "backup_entropy": False,
                "dsrl_u": {"enabled": enabled, "spec": copy.deepcopy(U_SPEC)},
            },
        }
    )
    worker.model = _SmallActorQ()
    worker.target_model = _SmallActorQ()
    worker.use_dsrl = True
    worker.use_dsrl_flat_replay = True
    worker._world_size = 1
    worker.critic_subsample_size = 0
    worker.torch_dtype = torch.float32
    worker.target_entropy = -16
    worker.entropy_temp = SimpleNamespace(
        alpha=torch.tensor(0.7),
        compute_alpha=lambda: torch.tensor(0.7),
    )
    worker._setup_dsrl_u_contract({"schema_version": 2 if enabled else 1})
    return worker


def _batch():
    return {
        "curr_obs": {"states": torch.arange(1, 5, dtype=torch.float32)[:, None]},
        "next_obs": {"states": torch.arange(2, 6, dtype=torch.float32)[:, None]},
        "actions": torch.ones(4, 1),
        "rewards": -torch.ones(4, 1),
        "continuations": torch.ones(4, 1, dtype=torch.bool),
        "terminations": torch.zeros(4, 1, dtype=torch.bool),
        "discounts": torch.full((4, 1), 0.999**10),
    }


def test_weighted_actor_gradient_and_microbatch_equivalence():
    worker = _worker()
    batch = _batch()
    raw = torch.tensor([0.01, 0.02, 0.2, 0.8])[:, None].expand(-1, 10)
    weights, _ = build_dsrl_u_weights(raw, torch.ones_like(raw, dtype=torch.bool))
    batch["dsrl_u_weight"] = weights
    full_loss, _, _ = inspect.unwrap(worker.forward_actor)(worker, batch)
    full_loss.backward()
    full_gradient = worker.model.theta.grad.clone()
    expected = ((0.7 * 2 - 3) * weights * batch["curr_obs"]["states"]).mean()
    torch.testing.assert_close(full_gradient, expected)

    worker.model.theta.grad = None
    for start in (0, 2):
        micro = {
            "curr_obs": {"states": batch["curr_obs"]["states"][start : start + 2]},
            "dsrl_u_weight": weights[start : start + 2],
        }
        loss, _, _ = inspect.unwrap(worker.forward_actor)(worker, micro)
        (loss / 2).backward()
    torch.testing.assert_close(worker.model.theta.grad, full_gradient)


def test_clean_actor_and_critic_alpha_objectives_remain_original():
    clean, weighted = _worker(False), _worker(True)
    batch = _batch()
    original, _, _ = inspect.unwrap(clean.forward_actor)(clean, batch)
    expected = ((0.7 * 2 - 3) * clean.model.theta * batch["curr_obs"]["states"]).mean()
    torch.testing.assert_close(original, expected)
    batch["dsrl_u_weight"] = torch.tensor([[1.3], [1.1], [0.9], [0.7]])
    clean_td, _ = inspect.unwrap(clean.forward_critic)(clean, batch)
    weighted_td, _ = inspect.unwrap(weighted.forward_critic)(weighted, batch)
    torch.testing.assert_close(clean_td, weighted_td)
    clean_alpha = inspect.unwrap(clean.forward_alpha)(clean, batch)
    weighted_alpha = inspect.unwrap(weighted.forward_alpha)(weighted, batch)
    torch.testing.assert_close(clean_alpha, weighted_alpha)


def test_u_contract_rejects_collection_mismatch_and_multi_rank():
    worker = _worker()
    worker.cfg.actor.model.openpi.dsrl_u_enabled = False
    with pytest.raises(ValueError, match="must agree"):
        worker._setup_dsrl_u_contract({"schema_version": 2})
    worker.cfg.actor.model.openpi.dsrl_u_enabled = True
    worker._world_size = 2
    with pytest.raises(ValueError, match="world_size=1"):
        worker._setup_dsrl_u_contract({"schema_version": 2})


def test_u_contract_does_not_coerce_invalid_yaml_scalar_types():
    worker = _worker()
    worker.cfg.algorithm.dsrl_u.enabled = "false"
    with pytest.raises(ValueError, match="booleans"):
        worker._setup_dsrl_u_contract({"schema_version": 2})
    worker.cfg.algorithm.dsrl_u.enabled = True
    for invalid_temperature in (True, "2.5"):
        worker.cfg.algorithm.dsrl_u.temperature = invalid_temperature
        with pytest.raises(ValueError, match="positive number"):
            worker._setup_dsrl_u_contract({"schema_version": 2})


def test_u_actor_rejects_missing_or_differentiable_weight():
    worker, batch = _worker(), _batch()
    with pytest.raises(ValueError, match="detached positive"):
        inspect.unwrap(worker.forward_actor)(worker, batch)
    batch["dsrl_u_weight"] = torch.ones(4, 1, requires_grad=True)
    with pytest.raises(ValueError, match="detached positive"):
        inspect.unwrap(worker.forward_actor)(worker, batch)
