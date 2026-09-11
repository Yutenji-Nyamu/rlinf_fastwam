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
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rabc_actor_contract_algorithm", ROOT / "rlinf/algorithms/online_bc_rabc.py"
)
ALGORITHM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ALGORITHM
SPEC.loader.exec_module(ALGORITHM)
DATA_SPEC = importlib.util.spec_from_file_location("rabc_actor_data", ROOT / "rlinf/data/online_bc.py")
DATA = importlib.util.module_from_spec(DATA_SPEC)
sys.modules[DATA_SPEC.name] = DATA
DATA_SPEC.loader.exec_module(DATA)


def split_batch(data, count):
    if isinstance(data, dict):
        parts = {key: split_batch(value, count) for key, value in data.items()}
        return [{key: value[i] for key, value in parts.items()} for i in range(count)]
    return list(data.chunk(count))


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
                 "normalized_batch_weights": ALGORITHM.normalized_batch_weights,
                 "mix_normalized_batch_weights": ALGORITHM.mix_normalized_batch_weights,
                 "np": np, "split_dict_to_chunk": split_batch,
                 "put_tensor_device": lambda batch, device: batch,
                 "append_to_dict": lambda metrics, items: [metrics.setdefault(k, []).append(v)
                                                           for k, v in items.items()]}
    exec(compile(ast.fix_missing_locations(extracted), str(path), "exec"), namespace)
    return namespace["ActualMethods"]


Actor = extract_class(
    ROOT / "rlinf/workers/actor/fsdp_online_bc_policy_worker.py",
    "EmbodiedOnlineBCFSDPPolicy",
    {"prepare_replay_batch", "rabc_identity", "validate_rabc_record", "load_checkpoint",
     "validate_rabc_checkpoint_identity", "reset_rabc_round_metrics"},
)
Dagger = extract_class(
    ROOT / "rlinf/workers/actor/fsdp_dagger_policy_worker.py",
    "EmbodiedDAGGERFSDPPolicy", {"update_buffer_one_epoch", "run_training"},
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

    def is_ready(self, minimum):
        return self.sample_batch is not None


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def actor(tmp_path, records=None):
    result = Actor()
    result.cfg = Config(
        actor=Config(global_batch_size=1024, micro_batch_size=32,
                     optim=Config(clip_grad=1.0)),
        algorithm=Config(online_bc=Config(data_path=str(tmp_path)),
                         replay_buffer=Config(min_buffer_size=1), update_epoch=2),
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
    result.is_weight_offloaded = False
    result.is_optimizer_offloaded = False
    result.device = torch.device("cpu")
    result.strategy_loads = []
    result._strategy = SimpleNamespace(load_checkpoint=lambda **kwargs: result.strategy_loads.append(kwargs))
    return result


def batch(ctx, deltas):
    return {"forward_inputs": {
        "rabc_delta_seconds": torch.as_tensor(deltas, dtype=torch.float64),
        "rabc_identity": torch.tensor(list(bytes.fromhex(ctx.rabc_client.identity_hash)), dtype=torch.uint8)
                              .expand(len(deltas), -1).clone(),
        "action_valid_mask": torch.ones((len(deltas), 50, 14), dtype=torch.bool),
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


@pytest.mark.parametrize("clean_mix", [0.0, 0.5, 1.0])
def test_actual_batch_transform_and_masked_loss_match_mixture(tmp_path, clean_mix):
    ctx = actor(tmp_path)
    ctx.rabc_config = ALGORITHM.RABCConfig(2.0, clean_mix=clean_mix)
    deltas = torch.cat([torch.full((256,), value) for value in (-1.0, 0.0, 1.0, 4.0)]).double()
    inputs = batch(ctx, deltas)
    mask = inputs["forward_inputs"]["action_valid_mask"]
    mask[::2, 25:] = False
    ctx.rabc_debug_remaining = 1
    actual = ctx.prepare_replay_batch(inputs)["forward_inputs"]["sample_weights"]
    raw = ALGORITHM.raw_weights(deltas, ctx.rabc_stats, ctx.rabc_config)
    normalized, _ = ALGORITHM.normalized_batch_weights(raw, ctx.rabc_config)
    target = torch.linspace(-1, 2, 1024 * 50 * 14, dtype=torch.float64).reshape(1024, 50, 14)
    target[~mask] = 1e6  # Finite padding must contribute neither loss nor gradient.
    parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    errors = (parameter - target).square()
    expected = (clean_mix * DATA.masked_fm_loss(errors, mask)
                + (1 - clean_mix) * DATA.masked_fm_loss(errors, mask, normalized))
    expected_grad = torch.autograd.grad(expected, parameter)[0]
    parameter2 = parameter.detach().clone().requires_grad_()
    actual_loss = sum(DATA.masked_fm_loss((parameter2 - t).square(), m, w) / 32
                      for t, m, w in zip(target.split(32), mask.split(32), actual.split(32)))
    actual_loss.backward()
    torch.testing.assert_close(actual_loss.detach(), expected.detach(), rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(parameter2.grad, expected_grad, rtol=1e-7, atol=1e-7)
    debug = torch.load(next((tmp_path / "weight_debug").glob("*.pt")), weights_only=True)
    torch.testing.assert_close(debug["normalized_weights"], normalized)
    torch.testing.assert_close(debug["sample_weights"].float(), actual)
    assert debug["identity"]["clean_mix"] == clean_mix


def test_partial_zero_micro_does_not_trigger_full_batch_fallback(tmp_path):
    ctx = actor(tmp_path)
    ctx.rabc_config = ALGORITHM.RABCConfig(2.0, clean_mix=0.5)
    actual = ctx.prepare_replay_batch(batch(ctx, [-1.0] * 32 + [4.0] * 992))
    weights = actual["forward_inputs"]["sample_weights"]
    torch.testing.assert_close(weights[:32], torch.full((32,), 0.5))
    assert ctx.rabc_metrics["rabc/clean_fallback"] == 0
    assert ctx.rabc_metrics["rabc/zero_fraction"] == 32 / 1024
    assert ctx.rabc_metrics["rabc/final_weight_zero_fraction"] == 0


@pytest.mark.parametrize("fault", ["empty_mask", "bad_mask_dtype", "nan_score", "bad_identity"])
def test_fallback_does_not_hide_invalid_data(tmp_path, fault):
    ctx = actor(tmp_path)
    ctx.rabc_config = ALGORITHM.RABCConfig(2.0, clean_mix=0.5)
    data = batch(ctx, [-1.0] * 1024)
    inputs = data["forward_inputs"]
    if fault == "empty_mask":
        inputs["action_valid_mask"][0] = False
    elif fault == "bad_mask_dtype":
        inputs["action_valid_mask"] = inputs["action_valid_mask"].float()
    elif fault == "nan_score":
        inputs["rabc_delta_seconds"][0] = float("nan")
    else:
        inputs["rabc_identity"][0, 0] = 1
    with pytest.raises(ValueError):
        ctx.prepare_replay_batch(data)
    assert "rabc/clean_fallback" not in ctx.rabc_metrics


@pytest.mark.parametrize("clean_mix,expected_updates", [(0.0, 0), (0.5, 2)])
def test_actual_update_loop_zero_batch_adam_scheduler_and_learner(tmp_path, monkeypatch,
                                                                clean_mix, expected_updates):
    ctx = actor(tmp_path)
    ctx.rabc_config = ALGORITHM.RABCConfig(2.0, clean_mix=clean_mix)
    ctx.replay_buffer.sample_batch = batch(ctx, [-1.0] * 1024)
    ctx.enable_online_lerobot = False
    ctx.enable_drq = False
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    model.weight.data.fill_(0.25)
    model.clip_grad_norm_ = lambda max_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    ctx.model = model
    ctx.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(ctx.optimizer, lambda step: 1.0)
    ctx.amp_context = nullcontext()
    ctx.before_micro_batch = lambda *args, **kwargs: nullcontext()
    ctx.grad_scaler = SimpleNamespace(scale=lambda loss: loss)
    ctx.forward_actor = lambda data: DATA.masked_fm_loss(
        (model.weight.reshape(1, 1, 1).expand(32, 50, 14) - 1).square(),
        data["action_valid_mask"], data["sample_weights"])
    ctx.update_one_epoch = lambda: Dagger.update_buffer_one_epoch(ctx)
    ctx.process_train_metrics = lambda metrics: metrics
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    ctx.reset_rabc_round_metrics()
    Dagger.run_training(ctx)
    assert ctx.update_step == expected_updates
    assert ctx.lr_scheduler.last_epoch == expected_updates
    assert len(ctx.optimizer.state) == int(expected_updates > 0)
    assert ctx.rabc_metrics["rabc/clean_fallback_updates_this_round"] == expected_updates
    assert ctx.rabc_metrics["rabc/skipped_updates_this_round"] == 2 - expected_updates
    if expected_updates:
        assert ctx.rabc_metrics["rabc/final_effective_sample_size"] == 1024
        assert model.weight.item() > 0.25
    else:
        assert model.weight.item() == 0.25
    ctx.reset_rabc_round_metrics()
    assert ctx.rabc_metrics["rabc/clean_fallback_updates_this_round"] == 0
    assert ctx.rabc_metrics["rabc/skipped_updates_this_round"] == 0


@pytest.mark.parametrize("offload", [False, True])
def test_restore_device_load_precedes_strategy_and_same_weights(tmp_path, offload):
    ctx = actor(tmp_path)
    ctx.rabc_config = ALGORITHM.RABCConfig(2.0, clean_mix=0.5)
    ctx.replay_buffer.records = [record(ctx, 8, 6, 0), record(ctx, 6, 5, 1)]
    ctx.rabc_stats = ALGORITHM.RABCStats()
    ctx.rabc_stats.update([2.0, 1.0])
    before = ctx.prepare_replay_batch(batch(ctx, [1.0, 2.0] * 512))["forward_inputs"]["sample_weights"]
    checkpoint = tmp_path / "checkpoint"
    save_method_checkpoint(ctx, checkpoint)
    events = []
    ctx.is_weight_offloaded = ctx.is_optimizer_offloaded = offload
    ctx.load_param_and_grad = lambda device: events.append("parameters")
    ctx.load_optimizer = lambda device: events.append("optimizer")

    def load_strategy(**kwargs):
        assert not ctx.is_weight_offloaded and not ctx.is_optimizer_offloaded
        events.append("strategy")

    ctx._strategy.load_checkpoint = load_strategy
    ctx.rabc_stats = ALGORITHM.RABCStats()
    ctx.load_checkpoint(checkpoint)
    after = ctx.prepare_replay_batch(batch(ctx, [1.0, 2.0] * 512))["forward_inputs"]["sample_weights"]
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert events == (["parameters", "optimizer", "strategy"] if offload else ["strategy"])
    assert ctx.update_step == 7


def test_legacy_identity_only_accepts_exact_old_schema_with_mix_zero(tmp_path):
    ctx = actor(tmp_path)
    legacy = {k: v for k, v in ctx.rabc_identity().items()
              if k not in ("clean_mix", "all_zero_policy", "weight_postprocess")}
    legacy["version"] = 1
    assert ctx.validate_rabc_checkpoint_identity(legacy)
    for fault in ({**legacy, "unexpected": 0}, {**legacy, "version": True},
                  {**legacy, "epsilon_weight": 1e-4}):
        with pytest.raises(ValueError):
            ctx.validate_rabc_checkpoint_identity(fault)
    checkpoint = tmp_path / "checkpoint"
    target, state = save_method_checkpoint(ctx, checkpoint)
    state["identity"] = legacy
    torch.save(state, target / "rabc.pt")
    ctx.load_checkpoint(checkpoint)
    assert ctx.rabc_metrics["rabc/resume_legacy_identity"] == 1
    ctx.rabc_config = ALGORITHM.RABCConfig(2.0, clean_mix=0.5)
    with pytest.raises(ValueError, match="clean_mix=0"):
        ctx.validate_rabc_checkpoint_identity(legacy)
    current = ctx.rabc_identity()
    assert not ctx.validate_rabc_checkpoint_identity(current)
    for fault in ({**current, "clean_mix": 0.0}, {**current, "all_zero_policy": "skip"},
                  {**current, "version": 3}, {**current, "weight_postprocess": "before"}):
        with pytest.raises(ValueError):
            ctx.validate_rabc_checkpoint_identity(fault)
