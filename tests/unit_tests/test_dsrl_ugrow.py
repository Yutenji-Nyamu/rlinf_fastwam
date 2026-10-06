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

"""CPU math and actual sampler orchestration without loading VLA weights."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "rlinf/algorithms/dsrl_ugrow.py"
spec = importlib.util.spec_from_file_location("dsrl_ugrow_test_helper", HELPER)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def test_signal_spec_roundtrip_and_changed_definition_rejection():
    assert u.validate_signal_spec(json.loads(json.dumps(u.U_SPEC))) == u.U_SPEC
    for field, value in (
        ("main_steps", 4),
        ("side_steps", 3),
        ("mask", "executed_actions"),
        ("schema_version", True),
        ("epsilon", 0),
    ):
        changed = dict(u.U_SPEC, **{field: value})
        with pytest.raises(ValueError):
            u.validate_signal_spec(changed)
    with pytest.raises(ValueError):
        u.validate_signal_spec(dict(u.U_SPEC, extra=True))


def test_disagreement_matches_numpy_population_std_rms():
    np = pytest.importorskip("numpy")
    a = torch.tensor([[[0.0, 1.0, -2.0], [1e-9, 2.0, -3.0]]])
    b = torch.tensor([[[0.0, -1.0, -2.0], [-1e-9, 2.3, 7.0]]])
    result, valid = u.relative_disagreement(a.requires_grad_(), b)
    pair = np.stack([a.detach().numpy(), b.numpy()]).astype(np.float64)
    expected = (pair.std(axis=0) / (np.sqrt((pair**2).mean(axis=0)) + 1e-8)).mean(
        axis=-1
    )
    torch.testing.assert_close(result, torch.tensor(expected).float())
    assert valid.all() and not result.requires_grad
    zeros, zeros_valid = u.relative_disagreement(
        torch.zeros_like(a), torch.zeros_like(b)
    )
    assert torch.equal(zeros, torch.zeros_like(zeros)) and zeros_valid.all()
    with pytest.raises(ValueError, match="nonfinite"):
        u.relative_disagreement(a * float("nan"), b)


def test_mean_log_order_mask_and_temperature_divisor():
    # A has a larger arithmetic mean but a smaller geometric mean than B.
    values = torch.tensor([[0.0001, 1.0, float("nan")], [0.02, 0.02, -1.0]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    weights, metrics = u.build_dsrl_u_weights(values, valid, temperature=2.5)
    assert weights.shape == (2, 1) and weights[0] < weights[1]
    torch.testing.assert_close(
        weights[1] / weights[0], torch.exp(torch.tensor([1 / 2.5]))
    )
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    assert 0 < metrics["weight_ess_fraction"] <= 1


@pytest.mark.parametrize("constant", [0.0, 0.2, 1.0])
def test_constant_scores_are_uniform(constant):
    values = torch.full((7, 10), constant, requires_grad=True)
    weights, _ = u.build_dsrl_u_weights(
        values, torch.ones_like(values, dtype=torch.bool)
    )
    assert torch.equal(weights, torch.ones(7, 1)) and not weights.requires_grad


def test_invalid_replay_signal_fails_closed():
    valid = torch.ones(2, 3, dtype=torch.bool)
    for value in [float("nan"), float("inf"), -0.1]:
        raw = torch.ones(2, 3)
        raw[0, 0] = value
        with pytest.raises(ValueError, match="finite and nonnegative"):
            u.build_dsrl_u_weights(raw, valid)
    with pytest.raises(ValueError, match="at least one"):
        u.build_dsrl_u_weights(torch.ones(2, 3), torch.zeros_like(valid))
    with pytest.raises(ValueError):
        u.build_dsrl_u_weights(torch.ones(2, 3), valid.float())


def test_global_weights_preserve_microbatch_gradient_and_neutral_baseline():
    raw = torch.arange(1, 81, dtype=torch.float32).reshape(8, 10) / 80
    weights, _ = u.build_dsrl_u_weights(raw, torch.ones_like(raw, dtype=torch.bool))
    features = torch.arange(1, 9, dtype=torch.float32).unsqueeze(-1)
    full_parameter = torch.tensor(0.3, requires_grad=True)
    full_loss = (weights * ((full_parameter * features).square() - features)).mean()
    full_loss.backward()
    micro_parameter = torch.tensor(0.3, requires_grad=True)
    for index in range(0, 8, 2):
        loss = (
            weights[index : index + 2]
            * (
                (micro_parameter * features[index : index + 2]).square()
                - features[index : index + 2]
            )
        ).mean() / 4
        loss.backward()
    torch.testing.assert_close(full_parameter.grad, micro_parameter.grad)
    baseline = ((full_parameter * features).square() - features).mean()
    neutral, _ = u.build_dsrl_u_weights(
        torch.zeros_like(raw), torch.ones_like(raw, dtype=torch.bool)
    )
    torch.testing.assert_close(
        baseline, (neutral * ((full_parameter * features).square() - features)).mean()
    )


def _actual_method(name):
    """Load the real method body; stub only unavailable model dependencies."""
    source = ROOT / "rlinf/models/embodiment/openpi/openpi_action_model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node = copy.deepcopy(node)
    for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        argument.annotation = None
    node.returns = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {
        "torch": torch,
        "math": __import__("math"),
        "relative_disagreement": u.relative_disagreement,
        "_model": SimpleNamespace(
            Observation=SimpleNamespace(from_dict=lambda data: SimpleNamespace(**data))
        ),
        "copy_dict_tensor": lambda data: {
            key: value.clone() for key, value in data.items()
        },
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


class _FakeSampler:
    sample_actions = _actual_method("sample_actions")

    def __init__(self):
        self.config = SimpleNamespace(
            use_dsrl=True,
            dsrl_u_enabled=True,
            num_steps=10,
            action_horizon=50,
            action_dim=32,
        )
        self.dsrl_u_spec = dict(u.U_SPEC)
        self.action_in_proj = SimpleNamespace(
            weight=torch.empty(1, dtype=torch.bfloat16)
        )
        self.calls = []
        self.prefix_calls = 0
        self.prefix = object()

    def _preprocess_observation(self, observation, train=False):
        assert not train
        return None, None, None, None, observation.state

    def _build_prefix_cache(self, *args):
        self.prefix_calls += 1
        return self.prefix, self.prefix, self.prefix

    def _sample_actions_with_prefix_cache(
        self, state, prefix, mask, cache, *, noise, mode, compute_values, num_steps=None
    ):
        assert prefix is mask is cache is self.prefix
        assert mode == "eval" and not torch.is_grad_enabled()
        steps = 10 if num_steps is None else num_steps
        self.calls.append((steps, noise.clone(), compute_values))
        # Match the real ODE sampler's otherwise unused random draws.
        for _ in range(steps):
            torch.randn_like(noise)
        return {"actions": noise.float() + 1.0 / steps, "chains": noise.clone()}


def test_actual_sampler_sidechain_reuses_cast_noise_and_preserves_rng():
    observation = SimpleNamespace(state=torch.zeros(2, 14))
    initial = torch.linspace(-0.9999, 1.0001, 2 * 50 * 32).reshape(2, 50, 32)
    initial_copy = initial.clone()
    before = torch.random.get_rng_state()
    clean = _FakeSampler()
    main = clean.sample_actions(observation, noise=initial, mode="eval")
    after_main = torch.random.get_rng_state()
    torch.random.set_rng_state(before)
    weighted = _FakeSampler()
    result = weighted.sample_actions(
        observation, noise=initial, mode="eval", collect_dsrl_u=True
    )
    assert torch.equal(after_main, torch.random.get_rng_state())
    assert torch.equal(initial, initial_copy)
    assert weighted.prefix_calls == clean.prefix_calls == 1
    assert [call[0] for call in weighted.calls] == [10, 5]
    assert weighted.calls[1][2] is False
    assert weighted.calls[0][1].dtype == torch.bfloat16
    assert torch.equal(weighted.calls[0][1], weighted.calls[1][1])
    for key in main:
        assert torch.equal(main[key], result[key])
    assert result["dsrl_u"].shape == result["dsrl_u_valid"].shape == (2, 10)
    assert result["dsrl_u_valid"].all()


def test_actual_sampler_rejects_signal_on_training_sde_path():
    sampler = _FakeSampler()
    with pytest.raises(ValueError, match="ODE inference"):
        sampler.sample_actions(
            SimpleNamespace(state=torch.zeros(2, 14)),
            noise=torch.ones(2, 50, 32),
            mode="train",
            collect_dsrl_u=True,
        )


class _FakePrediction(_FakeSampler):
    predict_action_batch = _actual_method("predict_action_batch")

    def __init__(self, phase, enabled):
        super().__init__()
        self.config.dsrl_u_enabled = enabled
        self.config.dsrl_action_noise_dim = 32
        self.config.is_nft = False
        self.dsrl_policy_phase = torch.tensor(phase, dtype=torch.int64)
        self.dsrl_action_noise_net = torch.nn.Linear(1, 1).to(torch.bfloat16)
        self.global_step = 17

    def obs_processor(self, env_obs):
        return {"states": env_obs["states"]}

    def input_transform(self, obs, transpose=False):
        return {
            "state": obs["states"],
            "tokenized_prompt": torch.ones(2, 4, dtype=torch.int64),
            "tokenized_prompt_mask": torch.ones(2, 4, dtype=torch.bool),
        }

    def precision_processor(self, obs):
        return obs

    def sac_forward(self, obs, train=False, mode="train"):
        latent = torch.randn(2, 32).tanh().to(torch.bfloat16)
        return latent[:, None].repeat(1, 50, 1), latent.float().sum(-1), None

    def output_transform(self, outputs):
        return {"actions": outputs["actions"][:, :10, :14]}

    def _sample_actions_with_prefix_cache(self, *args, **kwargs):
        result = super()._sample_actions_with_prefix_cache(*args, **kwargs)
        result["denoise_inds"] = torch.full((2, 10), -1)
        return result


@pytest.mark.parametrize("phase", [0, 1])
def test_actual_predict_preserves_behavior_and_records_only_training(phase):
    clean = _FakePrediction(phase, enabled=False)
    weighted = _FakePrediction(phase, enabled=True)
    observation = {"states": torch.zeros(2, 14), "main_images": torch.zeros(2, 3, 8, 8)}
    before = torch.random.get_rng_state()
    main_actions, main_info = clean.predict_action_batch(observation, mode="train")
    after_main = torch.random.get_rng_state()
    torch.random.set_rng_state(before)
    actions, info = weighted.predict_action_batch(observation, mode="train")
    assert torch.equal(after_main, torch.random.get_rng_state())
    assert torch.equal(actions, main_actions)
    assert torch.equal(info["prev_logprobs"], main_info["prev_logprobs"])
    assert torch.equal(
        info["forward_inputs"]["action"], main_info["forward_inputs"]["action"]
    )
    for key, expected in (("dsrl_u_policy_step", 17), ("dsrl_u_phase", phase)):
        assert torch.equal(info["forward_inputs"][key], torch.full((2, 1), expected))
    assert info["forward_inputs"]["dsrl_u"].shape == (2, 10)
    weighted.calls.clear()
    _, evaluation = weighted.predict_action_batch(observation, mode="eval")
    assert [call[0] for call in weighted.calls] == [10]
    assert "dsrl_u" not in evaluation["forward_inputs"]
