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


"""Targeted checks of actual sampler and actor methods; run on the server."""

import ast
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_dvac_adv_new_actor import expected_weights, fake_all_gather, make_actor

from rlinf.algorithms.grpo_signals import (
    capture_expert_norm,
    compute_ugrow_signal,
    grpo_signal_contract,
    grpo_signal_key,
)


def sampler_class():
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/openpi/openpi_action_model.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "OpenPi0ForRLActionPrediction"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_sample_actions_with_prefix_cache", "_get_timesteps"}
    ]
    node = ast.ClassDef(
        name="Sampler", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    ns = {
        "torch": torch,
        "np": np,
        "random": random,
        "nullcontext": nullcontext,
        "capture_expert_norm": capture_expert_norm,
        "compute_ugrow_signal": compute_ugrow_signal,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    return ns["Sampler"]


def make_sampler():
    obj = sampler_class()()
    obj.config = SimpleNamespace(
        num_steps=10,
        action_horizon=50,
        action_dim=32,
        action_env_dim=14,
        action_chunk=50,
        noise_method="flow_sde",
        joint_logprob=False,
        ignore_last=False,
        is_nft=False,
    )
    obj.use_vlm_value = False
    obj.action_in_proj = SimpleNamespace(weight=torch.empty((), dtype=torch.float32))
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    obj.paligemma_with_expert = SimpleNamespace(
        gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=layers))
    )
    obj.calls = []
    obj.sample_noise = lambda shape, device: torch.randn(shape, device=device)
    obj._init_nft_state = lambda *args: None
    obj._update_nft_state = lambda *args: None
    obj.get_logprob_norm = lambda x, m, std: -(x - m).square()

    def sample(x, idx, state, masks, cache, method, steps, compute):
        hidden = x
        for layer in layers:
            hidden = layer(hidden)
        velocity = torch.tanh(hidden) + (idx + 1) / steps
        obj.calls.append((steps, idx, method))
        std = torch.full_like(x, 0.07 if method == "flow_sde" else 0)
        return x - velocity / steps, std, torch.zeros(x.shape[:2]), velocity

    obj.sample_mean_var_val = sample
    return obj


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_tail():
    return (random.random(), np.random.rand(), torch.rand(4))


@pytest.mark.parametrize("source", ["ugrow_10_5", "norm_tail5_layers3"])
@pytest.mark.parametrize("seed", [7, 42])
def test_actual_sampler_preserves_main_actions_logprobs_and_rng(source, seed):
    args = (torch.ones(2, 14), None, None, None)
    seed_all(seed)
    baseline = make_sampler()
    clean = baseline._sample_actions_with_prefix_cache(*args)
    expected = rng_tail()
    seed_all(seed)
    model = make_sampler()
    result = model._sample_actions_with_prefix_cache(*args, return_grpo_signal=source)
    actual = rng_tail()
    for key in clean:
        assert torch.equal(clean[key], result[key]), key
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    assert result["grpo_signal"].shape == (2, 50)
    assert torch.isfinite(result["grpo_signal"]).all()
    assert result["grpo_signal"].std() > 0
    assert not result["grpo_signal"].requires_grad
    assert len(model.calls) == (25 if source == "ugrow_10_5" else 10)
    assert model.calls[:10] == baseline.calls
    assert all(
        not layer._forward_hooks
        for layer in model.paligemma_with_expert.gemma_expert.model.layers
    )
    if source == "ugrow_10_5":
        assert all(method == "flow_ode" for _, _, method in model.calls[10:])


def test_norm_hook_reads_residual_l2_and_is_removed_on_error():
    m = make_sampler()
    h = torch.arange(2 * 50 * 32).float().reshape(2, 50, 32) / 100
    layers = m.paligemma_with_expert.gemma_expert.model.layers
    with capture_expert_norm(m, 50) as values:
        for layer in layers:
            h = layer(h)
    assert len(values) == 3
    for v in values:
        torch.testing.assert_close(v, torch.linalg.vector_norm(h, dim=-1))
    with pytest.raises(RuntimeError, match="deliberate"):
        with capture_expert_norm(m, 50):
            raise RuntimeError("deliberate")
    assert all(not layer._forward_hooks for layer in layers)


def test_u_math_and_padding():
    a = torch.zeros(2, 50, 32)
    b = torch.zeros_like(a)
    a[..., :14] = 1
    b[..., :14] = -1
    a[..., 14:] = float("nan")
    torch.testing.assert_close(compute_ugrow_signal(a, b), torch.ones(2, 50))
    assert torch.equal(compute_ugrow_signal(b, b), torch.zeros(2, 50))


@pytest.mark.parametrize("source", ["ugrow_10_5", "norm_tail5_layers3"])
@pytest.mark.parametrize("rank", [0, 1])
def test_actual_actor_uses_same_group_mapper_and_records_signal(
    monkeypatch, tmp_path, source, rank
):
    actor = make_actor(rank=rank, tmp_path=tmp_path, mapping="exp_mean")
    actor.dvac_train_cfg.update(
        mode="apply", application="chunk_clipped_action_advantage", signal_source=source
    )
    inputs = actor.rollout_batch["forward_inputs"]
    inputs["grpo_action_signal"] = inputs.pop("dvac_v_l3")
    fake_all_gather(monkeypatch, actor)
    actor._prepare_dvac_train_step()
    torch.testing.assert_close(
        inputs["dvac_weights"],
        expected_weights(mapping="exp_mean")[:, rank * 4 : (rank + 1) * 4],
    )
    assert actor._dvac_pending_step["config"]["signal"] == grpo_signal_contract(
        actor.dvac_train_cfg
    )


@pytest.mark.parametrize("source", ["ugrow_10_5", "norm_tail5_layers3"])
def test_real_actor_resume_rejects_other_signal_before_loading_model(tmp_path, source):
    actor = make_actor(tmp_path=tmp_path)
    actor.dvac_train_cfg.update(
        mode="apply", application="chunk_clipped_action_advantage", signal_source=source
    )
    actor.save_checkpoint(str(tmp_path), 7)
    same = make_actor(tmp_path=tmp_path)
    same.dvac_train_cfg = dict(actor.dvac_train_cfg)
    same.load_checkpoint(str(tmp_path))
    assert same.base_loaded == str(tmp_path)
    for other in [
        "dvca",
        "norm_tail5_layers3" if source == "ugrow_10_5" else "ugrow_10_5",
    ]:
        wrong = make_actor(tmp_path=tmp_path)
        wrong.dvac_train_cfg = dict(actor.dvac_train_cfg, signal_source=other)
        with pytest.raises(ValueError, match="resume configuration mismatch"):
            wrong.load_checkpoint(str(tmp_path))
        assert not hasattr(wrong, "base_loaded")


def test_legacy_contract_and_wrong_signal_configuration():
    assert grpo_signal_contract({}) == {}
    assert grpo_signal_key({}) == "dvac_v_l3"
    with pytest.raises(ValueError, match="Unknown"):
        grpo_signal_contract({"signal_source": "wrong"})
    with pytest.raises(ValueError, match="requires"):
        grpo_signal_contract({"signal_source": "ugrow_10_5", "mode": "apply"})


def test_graphics_scope_only_selects_gpu6_or_gpu7(monkeypatch):
    import importlib.util

    p = Path(__file__).resolve().parents[2] / "tools/grpo_un/graphics_scope_runtime.py"
    spec = importlib.util.spec_from_file_location("grpo_scope", p)
    scope = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scope)
    manifest = {
        "physical_gpus": [6, 7],
        "cards": {"6": {"gpu_uuid": "GPU-six"}, "7": {"gpu_uuid": "GPU-seven"}},
    }
    for raw, expected in [
        (None, 6),
        ("", 6),
        ("6", 6),
        ("7", 7),
        ("GPU-seven", 7),
        (scope.FULL_NODE_MASK, 6),
    ]:
        assert scope.select_target(manifest, raw) == expected
    for bad in ["0", "4", "5", "6,7"]:
        with pytest.raises(RuntimeError):
            scope.select_target(manifest, bad)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", scope.FULL_NODE_MASK)
    assert scope.check_cuda({"physical_gpu": 6, "gpu_uuid": "GPU-six"}) == "6"
