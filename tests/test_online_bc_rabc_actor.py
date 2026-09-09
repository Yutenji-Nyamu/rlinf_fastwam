"""Exercise actual actor method ASTs without importing Ray/FSDP policy setup."""

import ast
from contextlib import nullcontext
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rabc_actor_contract_algorithm", ROOT / "rlinf/algorithms/online_bc_rabc.py"
)
ALGORITHM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ALGORITHM
SPEC.loader.exec_module(ALGORITHM)


def extract_class(path, class_name, method_names):
    source = ast.parse(path.read_text())
    original = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = []
    for method in original.body:
        if isinstance(method, ast.FunctionDef) and method.name in method_names:
            method = copy.deepcopy(method)
            method.decorator_list = []
            selected.append(method)
    assert {method.name for method in selected} == set(method_names)
    extracted = ast.parse("class ActualMethods:\n    pass\n")
    extracted.body[0].body = selected
    namespace = {"torch": torch, "Path": Path, "RABCStats": ALGORITHM.RABCStats,
                 "raw_weights": ALGORITHM.raw_weights,
                 "normalized_batch_weights": ALGORITHM.normalized_batch_weights}
    exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), namespace)
    return namespace["ActualMethods"]


Actor = extract_class(
    ROOT / "rlinf/workers/actor/fsdp_online_bc_policy_worker.py",
    "EmbodiedOnlineBCFSDPPolicy",
    {"prepare_replay_batch", "rabc_identity", "validate_rabc_record", "load_checkpoint"},
)
Dagger = extract_class(
    ROOT / "rlinf/workers/actor/fsdp_dagger_policy_worker.py",
    "EmbodiedDAGGERFSDPPolicy", {"update_buffer_one_epoch"},
)


class Replay:
    max_success_chunks = 3

    def __init__(self, records=None):
        self.records = list(records or [])
        self.loads = 0
        self.sample_batch = None

    def load_checkpoint(self, path):
        self.loads += 1

    def sample(self, num_chunks):
        assert num_chunks == 1024
        return copy.deepcopy(self.sample_batch)


def actor(tmp_path, records=None):
    result = Actor()
    result.cfg = SimpleNamespace(
        actor=SimpleNamespace(global_batch_size=1024, micro_batch_size=32),
        algorithm=SimpleNamespace(online_bc=SimpleNamespace(data_path=str(tmp_path))),
    )
    result.rabc_enabled = True
    result.rabc_config = ALGORITHM.RABCConfig(kappa_seconds=2.0)
    result.rabc_stats = ALGORITHM.RABCStats()
    result.rabc_stats.update(torch.tensor([-1.0, 0.0, 1.0, 4.0], dtype=torch.float64))
    result.rabc_client = SimpleNamespace(identity={"scorer": "frozen-test"}, identity_hash="12" * 32)
    result.rabc_metrics = {}
    result.rabc_debug_remaining = 0
    result.rabc_debug_index = 0
    result.replay_buffer = Replay(records)
    result._rank = 0
    result._world_size = 1
    result.update_step = 0
    result.worker_timer = lambda name: nullcontext()
    result.model = object()
    result.optimizer = object()
    result.lr_scheduler = object()
    result.checkpoint_format = "local_shard"
    result.strategy_loads = []
    result._strategy = SimpleNamespace(load_checkpoint=lambda **kwargs: result.strategy_loads.append(kwargs))
    return result


def batch(ctx, deltas):
    return {"forward_inputs": {
        "rabc_delta_seconds": torch.as_tensor(deltas, dtype=torch.float64),
        "rabc_identity": torch.tensor(list(bytes.fromhex(ctx.rabc_client.identity_hash)), dtype=torch.uint8)
                              .expand(len(deltas), -1).clone(),
    }}


def record(ctx, before, after, index):
    return {"rabc_identity": torch.tensor(list(bytes.fromhex(ctx.rabc_client.identity_hash)), dtype=torch.uint8),
            "rabc_episode_key": torch.arange(16, dtype=torch.uint8),
            "query_idx": torch.tensor(index),
            "rabc_value_before": torch.tensor(before, dtype=torch.float64),
            "rabc_value_after": torch.tensor(after, dtype=torch.float64),
            "rabc_delta_seconds": torch.tensor(before - after, dtype=torch.float64)}


def save_method_checkpoint(ctx, checkpoint):
    target = checkpoint / "online_bc/rank_0"
    target.mkdir(parents=True)
    stats = ALGORITHM.RABCStats()
    if ctx.replay_buffer.records:
        stats.update(torch.stack([row["rabc_delta_seconds"] for row in ctx.replay_buffer.records]))
    state = {"identity": ctx.rabc_identity(), "stats": stats.state_dict(),
             "seen": sorted({bytes(row["rabc_episode_key"].tolist()).hex() for row in ctx.replay_buffer.records})}
    torch.save(state, target / "rabc.pt")
    torch.save({"update_step": 7}, target / "learner.pt")
    return target, state


def test_full_optimizer_batch_weights_keep_one_global_denominator(tmp_path):
    ctx = actor(tmp_path)
    # Unequal microbatch distributions make per-microbatch normalization wrong.
    deltas = torch.cat([torch.full((256,), value) for value in (-1.0, 0.0, 1.0, 4.0)]).double()
    actual = ctx.prepare_replay_batch(batch(ctx, deltas))["forward_inputs"]["sample_weights"]
    raw = ALGORITHM.raw_weights(deltas, ctx.rabc_stats, ctx.rabc_config)
    losses = torch.linspace(0.5, 5.0, 1024, dtype=torch.float64)
    expected_loss = (raw * losses).sum() / (raw.sum() + ctx.rabc_config.epsilon_weight)
    accumulated_loss = sum((weight.double() * loss).mean() / 32
                           for weight, loss in zip(actual.split(32), losses.split(32)))
    assert actual.shape == (1024,) and actual.dtype == torch.float32
    assert torch.allclose(accumulated_loss, expected_loss, atol=1e-7, rtol=1e-7)
    with pytest.raises(ValueError, match="complete optimizer batch"):
        ctx.prepare_replay_batch(batch(ctx, deltas[:32]))


def test_repeated_zero_batches_skip_adam_and_allow_two_debug_artifacts(tmp_path):
    ctx = actor(tmp_path)
    ctx.rabc_debug_remaining = 2
    ctx.replay_buffer.sample_batch = batch(ctx, [-1.0] * 1024)

    class UntouchableOptimizer:
        def __getattribute__(self, name):
            raise AssertionError(f"Zero-weight batch touched optimizer.{name}")

    ctx.optimizer = UntouchableOptimizer()
    for _ in range(2):
        assert Dagger.update_buffer_one_epoch(ctx) == {"optimizer/skipped": 1.0}
    debug_files = sorted((tmp_path / "weight_debug").glob("*.pt"))
    assert len(debug_files) == 2 and ctx.rabc_debug_remaining == 0
    assert ctx.update_step == 0
    assert all(torch.load(path, weights_only=True)["diagnostics"]["skip_update"] for path in debug_files)


def test_restore_rejects_identity_and_inconsistent_moments_before_model_load(tmp_path):
    for fault in ("identity", "moments"):
        ctx = actor(tmp_path / fault)
        ctx.replay_buffer.records = [record(ctx, 8.0, 6.0, 0), record(ctx, 6.0, 5.0, 1)]
        checkpoint = tmp_path / fault / "checkpoint"
        target, state = save_method_checkpoint(ctx, checkpoint)
        if fault == "identity":
            state["identity"]["kappa_seconds"] = 9.0
        else:
            # A plausible finite moment with the correct count must still fail.
            state["stats"]["raw_mean"] += 0.25
        torch.save(state, target / "rabc.pt")
        with pytest.raises(ValueError, match="identity|moments"):
            ctx.load_checkpoint(checkpoint)
        assert ctx.strategy_loads == []


def test_valid_and_empty_restore_preserve_stats_and_disable_repeated_debug(tmp_path):
    for empty in (False, True):
        ctx = actor(tmp_path / str(empty))
        ctx.replay_buffer.records = [] if empty else [record(ctx, 8.0, 6.0, 0), record(ctx, 6.0, 5.0, 1)]
        ctx.rabc_debug_remaining = 2
        checkpoint = tmp_path / str(empty) / "checkpoint"
        save_method_checkpoint(ctx, checkpoint)
        ctx.load_checkpoint(checkpoint)
        assert len(ctx.strategy_loads) == 1 and ctx.update_step == 7
        assert ctx.rabc_stats.count == (0 if empty else 2)
        assert ctx.rabc_stats.raw_mean == (0.0 if empty else 1.5)
        assert ctx.rabc_debug_remaining == 0
