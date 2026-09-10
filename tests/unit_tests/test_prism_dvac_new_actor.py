# Copyright 2025 The RLinf Authors.
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

"""CPU integration of actual combined actor methods without FSDP/Ray startup.

Only infrastructure dependencies are replaced. Quality, advantages, weights,
termination masks, shuffle and checkpoint methods use the source under test.
Execute in the source-locked server environment.
"""

import ast
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import rlinf.algorithms.dvac_two_level as helper_module
from rlinf.algorithms.dvac_rank_reward import trajectory_dvac_quality
from rlinf.algorithms.dvac_two_level import compute_dvac_two_level_weights
from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.utils.metric_utils import append_to_dict, compute_loss_mask
from rlinf.utils.nested_dict_process import (
    process_nested_dict_for_adv,
    process_nested_dict_for_train,
    split_dict_to_chunk,
)


class CheckpointBase:
    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.base_saved_step = step

    def load_checkpoint(self, load_path: str) -> None:
        self.base_load_count = getattr(self, "base_load_count", 0) + 1


def actor_harness() -> type:
    path = (
        Path(helper_module.__file__).resolve().parents[1]
        / "workers/actor/embodied_fsdp_actor_worker.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    actor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EmbodiedFSDPActor"
    )
    wanted = {
        "_validate_prism_dvac_composition",
        "_prism_dvac_contract",
        "_prepare_prism_dvac_advantage_inputs",
        "_process_received_rollout_batch",
        "compute_advantages_and_returns",
        "_dvac_two_level_contract",
        "_prism_dvac_composition_metrics",
        "_prepare_dvac_two_level_step",
        "_prepare_dvac_train_step",
        "_write_dvac_step_artifact",
        "run_training",
        "_dvac_sidecar_path",
        "save_checkpoint",
        "load_checkpoint",
    }
    methods = [
        node
        for node in actor.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in methods} == wanted
    for method in methods:
        method.decorator_list = [
            item for item in method.decorator_list if "no_grad" in ast.unparse(item)
        ]
    harness = ast.ClassDef(
        name="ActorHarness",
        bases=[ast.Name(id="CheckpointBase", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            harness,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "np": np,
        "Path": Path,
        "json": json,
        "os": os,
        "CheckpointBase": CheckpointBase,
        "trajectory_dvac_quality": trajectory_dvac_quality,
        "compute_dvac_two_level_weights": compute_dvac_two_level_weights,
        "calculate_adv_and_returns": calculate_adv_and_returns,
        "compute_loss_mask": compute_loss_mask,
        "compute_rollout_metrics": lambda batch: {},
        "process_nested_dict_for_adv": process_nested_dict_for_adv,
        "process_nested_dict_for_train": process_nested_dict_for_train,
        "split_dict_to_chunk": split_dict_to_chunk,
        "append_to_dict": append_to_dict,
        "all_reduce_dict": lambda values, **kwargs: values,
        "pop_critic_explained_variance_stats": lambda values: None,
        "clear_memory": lambda **kwargs: None,
    }
    exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), namespace)
    return namespace["ActorHarness"]


def make_actor(tmp_path: Path, prism: bool = True, dvac: bool = True):
    actor = actor_harness()()
    actor._rank, actor._world_size, actor.version = 0, 1, 1
    actor.device = torch.device("cpu")
    actor.prism_dvac_enabled = prism
    actor.prism_dvac_selected_l = actor.dvac_selected_l = 3
    actor.prism_dvac_lambda, actor.prism_dvac_log_eps = 0.2, 1e-12
    actor._prism_dvac_metrics = actor._prism_dvac_components = None
    actor.dvac_train_enabled = actor.dvac_two_level_enabled = dvac
    actor.dvac_train_mode = "apply" if dvac else "off"
    actor.dvac_train_application = "chunk_clipped_action_advantage"
    actor.dvac_train_cfg = {
        "normalization": "two_level_group",
        "scope": "both",
        "alpha_local": 1.0,
        "alpha_chunk": 1.0,
        "log_eps": 1e-12,
        "minmax_eps": 1e-6,
        "save_step_tensors": True,
    }
    actor.dvac_output_dir = tmp_path
    actor._dvac_pending_step = None
    actor.dvac_recent_stats = None
    actor._new_dvac_recent_stats = lambda: pytest.fail(
        "Historical statistics were touched"
    )
    actor.cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "algorithm": {
                "adv_type": "prism_rloo" if prism else "grpo",
                "filter_rewards": not prism,
                "normalize_advantages": False,
                "group_size": 8,
                "reward_type": "chunk_level",
                "logprob_type": "chunk_level",
                "loss_type": "actor",
                "loss_agg_func": "token-mean",
                "update_epoch": 2,
                "rewards_lower_bound": 0.1,
                "rewards_upper_bound": 0.9,
            },
            "actor": {
                "model": {"num_action_chunks": 3},
                "seed": 1234,
                "global_batch_size": 24,
                "micro_batch_size": 8,
            },
            "env": {
                "train": {
                    "max_episode_steps": 6,
                    "rollout_epoch": 1,
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
        }
    )
    # Three native G8 groups: all failure, all success and mixed.
    variance = (
        torch.arange(24)[None, :, None] / 20
        + torch.arange(2)[:, None, None] / 5
        + torch.arange(3)[None, None, :] / 10
    ).exp()
    variance[0, 0, 2] = 1e5
    variance[1, 0] = 1e6
    rewards = torch.zeros(2, 24, 3)
    rewards[1, 8:16, 2] = 1
    rewards[1, 16:20, 2] = 1
    dones = torch.zeros(3, 24, 3, dtype=torch.bool)
    dones[1, 0, 1:] = True
    dones[2, 0] = True
    dones[2, :, -1] = True
    actor.rollout_batch = actor._process_received_rollout_batch(
        {
            "rewards": rewards,
            "dones": dones,
            "prev_logprobs": torch.zeros(2, 24, 3, 14),
            "forward_inputs": {
                "dvac_v_l3": variance.clone().requires_grad_(),
                "query_ids": torch.arange(48).reshape(2, 24, 1),
            },
        }
    )
    actor._dvac_rollout_group_ids = torch.arange(24) // 8
    actor._validate_prism_dvac_composition()
    return actor


def local_gather(monkeypatch) -> list[torch.Tensor]:
    calls = []

    def gather(output, value):
        assert len(output) == 1
        output[0].copy_(value)
        calls.append(value.detach().clone())

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    return calls


def test_actual_prism_then_dvac_share_signal_and_preserve_two_masks(
    monkeypatch, tmp_path
):
    actor = make_actor(tmp_path)
    variance = actor.rollout_batch["forward_inputs"]["dvac_v_l3"]
    rewards = actor.rollout_batch["rewards"].clone()
    actor.compute_advantages_and_returns()
    assert actor.rollout_batch["forward_inputs"]["dvac_v_l3"] is variance
    components = actor._prism_dvac_components
    assert not components["action_mask"][0, 0, 2]
    assert actor.rollout_batch["loss_mask"][0, 0, 0]
    assert actor._prism_dvac_metrics[
        "prism_dvac/rescued_same_outcome_group_fraction"
    ] == pytest.approx(2 / 3)
    advantages = actor.rollout_batch["advantages"].clone()
    for section in (slice(0, 8), slice(8, 16)):
        assert (advantages[0, section] > 0).any()
        assert (advantages[0, section] < 0).any()
    calls = local_gather(monkeypatch)
    actor._prepare_dvac_train_step()
    assert (
        len(calls) == 9
    )  # Original six collectives plus three small diagnostic vectors.
    assert "dvac_v_l3" not in actor.rollout_batch["forward_inputs"]
    torch.testing.assert_close(actor.rollout_batch["rewards"], rewards)
    torch.testing.assert_close(actor.rollout_batch["advantages"], advantages)
    assert not actor.rollout_batch["forward_inputs"]["dvac_weights"].requires_grad
    pending = actor._dvac_pending_step
    assert pending["local_factors"][0, 0, 2] > pending["local_factors"][0, 0, 0]
    assert pending["metrics"]["prism_dvac_new/adv_composition_max_error"] < 1e-6
    for metric in pending["metrics"].values():
        assert np.isfinite(metric)
    assert sum(
        pending["metrics"][f"prism_dvac_new/{name}_weighted_adv_mass_share"]
        for name in ("all_failure", "all_success", "mixed")
    ) == pytest.approx(1.0)
    actor._write_dvac_step_artifact(pending)
    saved = torch.load(tmp_path / "runner_step_0001.pt", weights_only=False)
    for name, value in components.items():
        torch.testing.assert_close(saved["prism_components"][name], value)
    with pytest.raises(ValueError, match="Missing rollout DVAC signal"):
        actor._prepare_dvac_train_step()


def test_prism_only_keeps_original_consumption_and_new_only_keeps_filter(
    monkeypatch, tmp_path
):
    prism = make_actor(tmp_path, dvac=False)
    prism.compute_advantages_and_returns()
    assert "dvac_v_l3" not in prism.rollout_batch["forward_inputs"]
    assert prism._prism_dvac_components is None
    new = make_actor(tmp_path, prism=False)
    new.compute_advantages_and_returns()
    assert not new.rollout_batch["loss_mask"][:, :16].any()
    assert new.rollout_batch["loss_mask"][:, 16:].all()
    local_gather(monkeypatch)
    new._prepare_dvac_train_step()
    assert "prism_components" not in new._dvac_pending_step
    assert not any(
        name.startswith("prism") for name in new._dvac_pending_step["metrics"]
    )


def test_actual_training_keeps_weights_through_shuffle_and_u2_then_clears_state(
    monkeypatch, tmp_path
):
    actor = make_actor(tmp_path)
    actor.compute_advantages_and_returns()
    calls = local_gather(monkeypatch)
    actor.is_weight_offloaded = actor.is_optimizer_offloaded = False
    actor.model = torch.nn.Identity()
    actor.optimizer = SimpleNamespace(zero_grad=lambda: None)
    actor.lr_scheduler = SimpleNamespace(step=lambda: None)
    actor.torch_platform = SimpleNamespace(empty_cache=lambda: None)
    actor.gradient_accumulation = 3
    actor.optimizer_step = lambda: (torch.tensor(0.0), [5e-6])
    seen = []

    def micro_batch(micro_batch, metrics, **kwargs):
        assert "dvac_v_l3" not in micro_batch["forward_inputs"]
        seen.extend(
            zip(
                micro_batch["forward_inputs"]["query_ids"].reshape(-1).tolist(),
                micro_batch["forward_inputs"]["dvac_weights"].clone(),
            )
        )

    actor.train_micro_batch = micro_batch
    metrics = actor.run_training()
    assert len(calls) == 9
    assert Counter(query_id for query_id, _ in seen) == Counter(
        dict.fromkeys(range(48), 2)
    )
    saved = torch.load(tmp_path / "runner_step_0001.pt", weights_only=False)
    expected = saved["weights"].reshape(48, 3)
    for query_id, weight in seen:
        torch.testing.assert_close(weight, expected[query_id])
    assert actor._dvac_pending_step is None
    assert actor._prism_dvac_components is None
    assert actor._prism_dvac_metrics is None
    assert actor.dvac_recent_stats is None
    assert "prism_dvac/mixed_success_minus_failure_quality" in metrics
    assert "actor/dvac_weight_ess_fraction" in metrics


@pytest.mark.parametrize(
    "changed",
    [
        "quality_lambda",
        "selected_l",
        "log_eps",
        "adv_type",
        "filter_rewards",
        "normalize_advantages",
        "group_size",
        "alpha_chunk",
        "scope",
    ],
)
def test_joint_resume_rejects_changed_semantics_before_actor_load(tmp_path, changed):
    actor = make_actor(tmp_path)
    checkpoint = str(tmp_path / "checkpoint")
    actor.save_checkpoint(checkpoint, 7)
    payload = json.loads(actor._dvac_sidecar_path(checkpoint).read_text())
    assert payload["schema_version"] == 1
    assert payload["prism_dvac_config"] == actor._prism_dvac_contract()
    assert payload["recent_stats"] is None
    actor.load_checkpoint(checkpoint)
    assert actor.base_load_count == 1
    if changed == "quality_lambda":
        actor.prism_dvac_lambda = 0.3
    elif changed == "selected_l":
        actor.prism_dvac_selected_l = 4
    elif changed == "log_eps":
        actor.prism_dvac_log_eps = 1e-9
    elif changed == "alpha_chunk":
        actor.dvac_train_cfg[changed] = 0.5
    elif changed == "scope":
        actor.dvac_train_cfg[changed] = "positive"
    else:
        actor.cfg.algorithm[changed] = {
            "adv_type": "grpo",
            "filter_rewards": True,
            "normalize_advantages": True,
            "group_size": 4,
        }[changed]
    with pytest.raises(ValueError, match="resume configuration mismatch"):
        actor.load_checkpoint(checkpoint)
    assert actor.base_load_count == 1


@pytest.mark.parametrize("prism,dvac", [(False, True), (True, False), (False, False)])
def test_joint_checkpoint_cannot_silently_resume_as_a_single_method(
    tmp_path, prism, dvac
):
    checkpoint = str(tmp_path / "checkpoint")
    make_actor(tmp_path).save_checkpoint(checkpoint, 7)
    actor = make_actor(tmp_path, prism=prism, dvac=dvac)
    with pytest.raises(ValueError, match="disabled"):
        actor.load_checkpoint(checkpoint)
    assert not hasattr(actor, "base_load_count")


def test_old_new_only_schema_remains_compatible_but_is_not_joint_resume(tmp_path):
    checkpoint = str(tmp_path / "checkpoint")
    old = make_actor(tmp_path, prism=False)
    old.save_checkpoint(checkpoint, 7)
    payload = json.loads(old._dvac_sidecar_path(checkpoint).read_text())
    assert "prism_dvac_config" not in payload
    old.load_checkpoint(checkpoint)
    assert old.base_load_count == 1
    combined = make_actor(tmp_path)
    with pytest.raises(ValueError, match=r"Prism\+DVAC resume configuration mismatch"):
        combined.load_checkpoint(checkpoint)
    assert not hasattr(combined, "base_load_count")


def test_pending_prism_preparation_cannot_be_checkpointed(tmp_path):
    actor = make_actor(tmp_path)
    actor.compute_advantages_and_returns()
    assert actor._dvac_pending_step is None  # V has not yet been consumed by DVAC.
    with pytest.raises(RuntimeError, match="partially applied"):
        actor.save_checkpoint(str(tmp_path / "checkpoint"), 1)
    assert not (tmp_path / "checkpoint").exists()


@pytest.mark.parametrize(
    "changed", ["old_application", "selected_l", "log_eps", "filter", "adv_type"]
)
def test_actor_narrow_composition_guard(tmp_path, changed):
    actor = make_actor(tmp_path)
    if changed == "old_application":
        actor.dvac_train_application = "logprob_st"
    elif changed == "selected_l":
        actor.prism_dvac_selected_l = 4
    elif changed == "log_eps":
        actor.prism_dvac_log_eps = 1e-9
    elif changed == "filter":
        actor.cfg.algorithm.filter_rewards = True
    else:
        actor.cfg.algorithm.adv_type = "grpo"
    with pytest.raises(ValueError):
        actor._validate_prism_dvac_composition()
