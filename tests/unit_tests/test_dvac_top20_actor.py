"""Source-extracted real actor seams and CPU/Gloo two-rank integration.

Only scheduler decorators/base-class GPU setup are replaced. Top20 methods,
native batch loop, and checkpoint bodies are the production source verbatim.
This is not a substitute for the deferred GPU/FSDP smoke.
"""

import ast
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

import rlinf.algorithms.dvac_top20 as top20_module
from rlinf.algorithms.dvac_top20 import (
    DVACTop20Config, DVACTop20State, compute_top20_weights,
    distributed_top20_weights, expand_native_mask,
)
from rlinf.utils.nested_dict_process import process_nested_dict_for_train, split_dict_to_chunk


class CheckpointBase:
    def save_checkpoint(self, save_path, step=0):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.base_saved = step

    def load_checkpoint(self, load_path):
        self.base_loaded = str(load_path)


def append_metrics(metrics, entries):
    for key, value in entries.items():
        metrics.setdefault(key, []).append(value)


def actor_harness():
    path = Path(top20_module.__file__).resolve().parents[1] / "workers/actor/embodied_fsdp_actor_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EmbodiedFSDPActor")
    wanted = {
        "_prepare_dvac_top20_rollout", "_prepare_dvac_top20_batch",
        "_dvac_top20_sidecar_path", "_dvac_top20_resume_budget",
        "save_checkpoint", "load_checkpoint", "run_training",
    }
    methods = [node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in methods} == wanted
    for method in methods:
        method.decorator_list = []
    extracted = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.ClassDef(name="ActorHarness", bases=[ast.Name(id="CheckpointBase", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]),
    ], type_ignores=[])
    namespace = {
        "torch": torch, "np": np, "Path": Path, "json": json, "os": os,
        "CheckpointBase": CheckpointBase, "DVACTop20State": DVACTop20State,
        "distributed_top20_weights": distributed_top20_weights,
        "expand_native_mask": expand_native_mask, "append_to_dict": append_metrics,
        "process_nested_dict_for_train": process_nested_dict_for_train,
        "split_dict_to_chunk": split_dict_to_chunk,
        "clear_memory": lambda **kwargs: None,
        "pop_critic_explained_variance_stats": lambda metrics: None,
        "all_reduce_dict": lambda values, **kwargs: values,
    }
    exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), namespace)
    return namespace["ActorHarness"]


def make_actor(rank=0, world_size=1, p=.1, domain="batch"):
    actor = actor_harness()()
    actor._rank, actor._world_size = rank, world_size
    actor.version = 17
    actor.dvac_train_enabled = False
    actor.dvac_top20_cfg = DVACTop20Config.from_dict({
        "enabled": True, "selection_domain": domain, "full_update_probability": p,
    })
    actor.dvac_top20_enabled = True
    actor.dvac_top20_state = DVACTop20State(actor.dvac_top20_cfg)
    actor._dvac_top20_pending_update = False
    actor.dvac_top20_output_dir = None
    actor.cfg = OmegaConf.create({
        "actor": {"global_batch_size": 8 * world_size, "micro_batch_size": 2, "seed": 1234, "model": {"num_action_chunks": 50}},
        "algorithm": {"update_epoch": 2, "group_size": 8, "loss_type": "actor"},
        "env": {"train": {"rollout_epoch": 4}},
    })
    return actor


def sample_batch(rank):
    # Both actors have high/low V chunks; global top-k differs from local top-k.
    offsets = torch.tensor([0., 50., 100., 500., 200., 550., 250., 600.]) + rank * 20
    v = offsets[:, None] + torch.arange(50).float()[None, :]
    mask = torch.ones(8, 1, dtype=torch.bool)
    if rank == 1:
        mask[3] = False
    return {
        "dvac_top20_v": v,
        "dvac_top20_query_ids": torch.arange(8) * 2 + rank,
        "loss_mask": mask,
        "forward_inputs": {"native_input": torch.arange(8)[:, None]},
        "prev_logprobs": torch.zeros(8, 50, 14),
    }


def test_rollout_id_and_variance_follow_native_shuffle():
    actor = make_actor()
    v = torch.arange(8 * 50).reshape(2, 4, 50).float()
    actor.rollout_batch = {"prev_logprobs": torch.zeros(2, 4, 50, 14), "forward_inputs": {"dvac_v_l3": v.clone()}}
    actor._prepare_dvac_top20_rollout()
    assert "dvac_v_l3" not in actor.rollout_batch["forward_inputs"]
    order = torch.tensor([7, 0, 5, 2, 1, 6, 3, 4])
    shuffled = process_nested_dict_for_train(actor.rollout_batch, order)
    assert torch.equal(shuffled["dvac_top20_query_ids"], order)
    assert torch.equal(shuffled["dvac_top20_v"], v.reshape(8, 50)[order])


def test_checkpoint_exact_method_restore_and_mismatch_before_base(tmp_path):
    actor = make_actor(p=.3)
    for _ in range(11):
        actor.dvac_top20_state.next_full_update()
    actor.save_checkpoint(str(tmp_path / "checkpoint"), 17)
    expected = [actor.dvac_top20_state.next_full_update() for _ in range(30)]
    restored = make_actor(p=.3)
    restored.load_checkpoint(str(tmp_path / "checkpoint"))
    assert restored.base_loaded == str(tmp_path / "checkpoint")
    assert [restored.dvac_top20_state.next_full_update() for _ in range(30)] == expected
    mismatched = make_actor(p=.1)
    with pytest.raises(ValueError, match="identity"):
        mismatched.load_checkpoint(str(tmp_path / "checkpoint"))
    assert not hasattr(mismatched, "base_loaded")
    missing = make_actor()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        missing.load_checkpoint(str(tmp_path / "missing"))
    assert not hasattr(missing, "base_loaded")
    actor._dvac_top20_pending_update = True
    with pytest.raises(RuntimeError, match="partially"):
        actor.save_checkpoint(str(tmp_path / "partial"), 18)
    assert not (tmp_path / "partial").exists()


def test_native_training_loop_one_gate_per_adam_all_micro(monkeypatch):
    actor = make_actor(p=.5)
    actor.is_weight_offloaded = actor.is_optimizer_offloaded = False
    actor.model = SimpleNamespace(train=lambda: None)
    actor.torch_platform = SimpleNamespace(empty_cache=lambda: None)
    actor.gradient_accumulation = 4
    actor.lr_scheduler = SimpleNamespace(step=lambda: None)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    actor.optimizer = torch.optim.SGD([parameter], lr=.01)
    history = []
    optimizer_calls = []

    def train_micro_batch(micro_batch, metrics, is_last):
        history.append({
            "update_index": actor.dvac_top20_state.update_index,
            "weights": micro_batch["forward_inputs"]["dvac_top20_weights"].clone(),
            "ids": micro_batch["dvac_top20_query_ids"].clone(),
            "is_last": is_last,
        })
        parameter.square().backward()

    def optimizer_step():
        optimizer_calls.append(actor.dvac_top20_state.update_index)
        actor.optimizer.step()
        return 1., [.01]

    actor.train_micro_batch = train_micro_batch
    actor.optimizer_step = optimizer_step
    actor.rollout_batch = {
        "prev_logprobs": torch.zeros(2, 4, 50, 14),
        "advantages": torch.ones(2, 4, 1),
        "loss_mask": torch.ones(2, 4, 1, dtype=torch.bool),
        "forward_inputs": {"dvac_v_l3": torch.arange(400).reshape(2, 4, 50).float()},
    }
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    actor.run_training()
    assert optimizer_calls == [1, 2]
    assert len(history) == 8
    assert [record["update_index"] for record in history] == [1] * 4 + [2] * 4
    assert [record["is_last"] for record in history] == [False, False, False, True] * 2
    assert not actor._dvac_top20_pending_update
    reference = DVACTop20State(actor.dvac_top20_cfg)
    for update in range(2):
        full = reference.next_full_update()
        weights = torch.cat([entry["weights"] for entry in history[update * 4:(update + 1) * 4]])
        assert (weights > 0).sum() == (400 if full else 80)
        assert torch.allclose(weights.mean(), torch.tensor(1.))


def _gloo_worker(rank, world_size, init_file, output_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=90))
    try:
        reports = []
        for domain in ["chunk", "batch"]:
            for p in [0., 1., .3]:
                actor = make_actor(rank, world_size, p=p, domain=domain)
                reference = DVACTop20State(actor.dvac_top20_cfg)
                batch = sample_batch(rank)
                shards = [sample_batch(index) for index in range(world_size)]
                expected, _ = compute_top20_weights(
                    torch.cat([part["dvac_top20_v"] for part in shards]),
                    torch.cat([part["loss_mask"] for part in shards]),
                    torch.cat([part["dvac_top20_query_ids"] for part in shards]),
                    selection_domain=domain,
                )
                expected = expected[rank * 8:(rank + 1) * 8]
                for update in range(8):
                    actor._prepare_dvac_top20_batch(batch, {})
                    # Reference state uses same root-only decision protocol.
                    full = reference.next_full_update()
                    wanted = batch["loss_mask"].expand_as(expected).float() if full else expected
                    actual = batch["forward_inputs"]["dvac_top20_weights"]
                    assert torch.equal(actual, wanted)
                    for micro in split_dict_to_chunk(batch, 4):
                        assert micro["forward_inputs"]["dvac_top20_weights"].shape == (2, 50)
                    assert actor.dvac_top20_state.update_index == update + 1
                    reports.append([domain, p, update, full])
                    actor._dvac_top20_pending_update = False
                checkpoint = Path(output_dir) / f"{domain}_{p}"
                actor.save_checkpoint(str(checkpoint), 17)
                saved = make_actor(rank, world_size, p=p, domain=domain)
                saved.load_checkpoint(str(checkpoint))
                assert saved.dvac_top20_state.state_dict() == actor.dvac_top20_state.state_dict()
                for _ in range(3):
                    assert saved.dvac_top20_state.next_full_update() == actor.dvac_top20_state.next_full_update()
        Path(output_dir, f"rank{rank}.json").write_text(json.dumps(reports), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def test_cpu_gloo_two_rank_batch_selection_and_shared_resume(tmp_path):
    mp.spawn(_gloo_worker, args=(2, str(tmp_path / "gloo_init"), str(tmp_path)), nprocs=2, join=True)
    assert json.loads((tmp_path / "rank0.json").read_text()) == json.loads((tmp_path / "rank1.json").read_text())
