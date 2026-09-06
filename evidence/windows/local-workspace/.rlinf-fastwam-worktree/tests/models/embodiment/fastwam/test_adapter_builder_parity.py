# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight Fast-WAM RoboTwin adapter and policy-contract tests."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from rlinf.models.embodiment.fastwam.fastwam_policy import (
    FastWAMPolicy,
    FastWAMPolicyConfig,
)
from rlinf.models.embodiment.fastwam.robotwin_adapter import (
    ACTION_DIM,
    ACTION_HORIZON,
    DEFAULT_PROMPT,
    adapt_robotwin_observation,
    build_prompts,
    compose_three_camera_image,
    denormalize_actions,
    normalize_proprio,
    validate_env_obs,
)


class _MockActionNormalizer:
    def __init__(self) -> None:
        self.backward_calls = 0

    def backward(self, value: torch.Tensor) -> torch.Tensor:
        self.backward_calls += 1
        return value * 3.0 - 2.0


class _MockNormalizer:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.action = _MockActionNormalizer()
        self.normalizers = {"action": {"default": self.action}}

    def forward(self, batch: dict) -> dict:
        self.forward_calls += 1
        return {"state": {"default": batch["state"]["default"] * 2.0}}


class _MockProcessor:
    def __init__(self) -> None:
        self.shape_meta = {
            "state": [{"key": "default"}],
            "action": [{"key": "default"}],
        }
        self.normalizer = _MockNormalizer()
        self.transform_calls = 0

    def action_state_transform(self, batch: dict) -> dict:
        self.transform_calls += 1
        return {"state": {"default": batch["state"]["default"] + 1.0}}


def _env_obs(batch_size: int = 2) -> dict:
    main = torch.arange(batch_size * 7 * 9 * 3, dtype=torch.int64)
    main = (main % 256).to(torch.uint8).reshape(batch_size, 7, 9, 3)
    wrist = torch.arange(batch_size * 2 * 5 * 6 * 3, dtype=torch.int64)
    wrist = ((wrist * 7 + 19) % 256).to(torch.uint8)
    wrist = wrist.reshape(batch_size, 2, 5, 6, 3)
    states = torch.linspace(-1.0, 1.0, batch_size * ACTION_DIM).reshape(
        batch_size, ACTION_DIM
    )
    tasks = [f"perform task {index}" for index in range(batch_size)]
    return {
        "main_images": main,
        "wrist_images": wrist,
        "states": states,
        "task_descriptions": tasks,
    }


def _official_resize(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    return np.asarray(
        pil_image.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8
    )


def _official_composite(main: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    composites = []
    for index in range(main.shape[0]):
        head = _official_resize(main[index].numpy(), (320, 256))
        left = _official_resize(wrist[index, 0].numpy(), (160, 128))
        right = _official_resize(wrist[index, 1].numpy(), (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        composites.append(np.concatenate([head, bottom], axis=0))
    tensor = torch.from_numpy(np.stack(composites)).permute(0, 3, 1, 2)
    return tensor.to(torch.float32) * (2.0 / 255.0) - 1.0


def test_validate_env_obs_accepts_exact_robotwin_batch_contract():
    env_obs = _env_obs()

    batch = validate_env_obs(env_obs)

    assert batch.batch_size == 2
    assert batch.main_images is env_obs["main_images"]
    assert batch.wrist_images is env_obs["wrist_images"]
    assert batch.states is env_obs["states"]
    assert batch.task_descriptions == tuple(env_obs["task_descriptions"])


@pytest.mark.parametrize(
    ("field", "replacement", "error", "match"),
    [
        (
            "main_images",
            torch.zeros(2, 7, 9, 3),
            TypeError,
            "main_images must have dtype uint8",
        ),
        (
            "main_images",
            torch.zeros(2, 3, 7, 9, dtype=torch.uint8),
            ValueError,
            "main_images must have shape",
        ),
        (
            "wrist_images",
            torch.zeros(2, 1, 5, 6, 3, dtype=torch.uint8),
            ValueError,
            "wrist_images must have shape",
        ),
        (
            "states",
            torch.zeros(2, ACTION_DIM - 1),
            ValueError,
            "states must have shape",
        ),
        (
            "task_descriptions",
            ["only one"],
            ValueError,
            "length must equal batch size",
        ),
        (
            "task_descriptions",
            ["valid", "   "],
            ValueError,
            "must be non-empty",
        ),
    ],
)
def test_validate_env_obs_fails_fast(field, replacement, error, match):
    env_obs = _env_obs()
    env_obs[field] = replacement

    with pytest.raises(error, match=match):
        validate_env_obs(env_obs)


def test_validate_env_obs_rejects_missing_fields_and_nonfinite_state():
    missing = _env_obs()
    del missing["wrist_images"]
    with pytest.raises(KeyError, match="wrist_images"):
        validate_env_obs(missing)

    nonfinite = _env_obs()
    nonfinite["states"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_env_obs(nonfinite)


def test_three_camera_composition_matches_official_pil_bilinear_and_order():
    env_obs = _env_obs()

    actual = compose_three_camera_image(
        env_obs["main_images"],
        env_obs["wrist_images"],
        device="cpu",
        dtype=torch.float32,
    )
    expected = _official_composite(env_obs["main_images"], env_obs["wrist_images"])

    assert actual.shape == (2, 3, 384, 320)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_processor_mock_drives_state_normalization_and_action_denormalization():
    processor = _MockProcessor()
    states = _env_obs()["states"]

    normalized = normalize_proprio(states, processor)

    assert processor.transform_calls == 1
    assert processor.normalizer.forward_calls == 1
    assert normalized.device.type == "cpu"
    assert normalized.dtype == torch.float32
    torch.testing.assert_close(normalized, (states + 1.0) * 2.0)

    model_actions = torch.linspace(-1.0, 1.0, 2 * ACTION_HORIZON * ACTION_DIM).reshape(
        2, ACTION_HORIZON, ACTION_DIM
    )
    original_actions = model_actions.clone()
    physical = denormalize_actions(model_actions, processor)

    assert processor.normalizer.action.backward_calls == 1
    assert physical.device.type == "cpu"
    assert physical.dtype == torch.float32
    torch.testing.assert_close(physical, model_actions * 3.0 - 2.0)
    torch.testing.assert_close(model_actions, original_actions)


def test_processor_contract_rejects_nondefault_or_ambiguous_shape_meta():
    processor = _MockProcessor()
    processor.shape_meta["state"] = [{"key": "qpos"}]
    with pytest.raises(ValueError, match="requires.*default"):
        normalize_proprio(_env_obs()["states"], processor)

    processor = _MockProcessor()
    processor.shape_meta["action"].append({"key": "other"})
    actions = torch.zeros(1, ACTION_HORIZON, ACTION_DIM)
    with pytest.raises(ValueError, match="exactly one merged action key"):
        denormalize_actions(actions, processor)


def test_action_denormalization_rejects_noncanonical_shape():
    processor = _MockProcessor()
    with pytest.raises(ValueError, match="model_actions must have shape"):
        denormalize_actions(torch.zeros(ACTION_HORIZON, ACTION_DIM), processor)
    with pytest.raises(ValueError, match="model_actions must have shape"):
        denormalize_actions(torch.zeros(1, ACTION_HORIZON - 1, ACTION_DIM), processor)


def test_prompts_and_combined_adapter_preserve_batch_order():
    env_obs = _env_obs()
    processor = _MockProcessor()

    prompts = build_prompts(env_obs["task_descriptions"])
    image, proprio, adapted_prompts = adapt_robotwin_observation(
        env_obs,
        processor,
        device="cpu",
        dtype=torch.float32,
    )

    assert prompts == [
        DEFAULT_PROMPT.format(task=task) for task in env_obs["task_descriptions"]
    ]
    assert adapted_prompts == prompts
    assert image.shape == (2, 3, 384, 320)
    torch.testing.assert_close(proprio, (env_obs["states"] + 1.0) * 2.0)


def test_validation_does_not_copy_or_guess_missing_wrist_or_prompt():
    env_obs = deepcopy(_env_obs())
    env_obs["wrist_images"] = None
    with pytest.raises(TypeError, match="wrist_images must be a torch.Tensor"):
        validate_env_obs(env_obs)

    env_obs = _env_obs()
    env_obs["task_descriptions"] = None
    with pytest.raises(TypeError, match=r"task_descriptions must be a list\[str\]"):
        validate_env_obs(env_obs)


class _TinyMoT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixtures = nn.ModuleDict(
            {"action": nn.Linear(2, 2), "video": nn.Linear(2, 2)}
        )


class _TinyFastWAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mot = _TinyMoT()
        self.vae = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.proprio_encoder = nn.Linear(2, 2)
        self.torch_dtype = torch.float32


def _policy_replay(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "chains": torch.zeros(batch_size, 11, 32, 14),
        "denoise_inds": torch.zeros(batch_size, dtype=torch.long),
        "image": torch.zeros(batch_size, 3, 384, 320),
        "text_context": torch.zeros(batch_size, 128, 4096),
        "text_context_mask": torch.ones(batch_size, 128, dtype=torch.bool),
        "proprio": torch.zeros(batch_size, 14),
        "action": torch.zeros(batch_size, 24 * 14),
        "model_action": torch.zeros(batch_size, 32 * 14),
    }


def test_policy_module_mode_keeps_only_action_expert_in_train_mode():
    policy = FastWAMPolicy(
        model=_TinyFastWAM(),
        processor=object(),
        config=FastWAMPolicyConfig(),
    )
    assert policy.rlinf_accepts_rollout_mode is True
    assert not hasattr(policy, "value_head")

    policy.train(True)
    assert policy.training
    assert policy.model.training
    assert policy.model.mot.training
    assert policy.model.mot.mixtures["action"].training
    assert not policy.model.mot.mixtures["video"].training
    assert not policy.model.vae.training
    assert not policy.model.text_encoder.training
    assert not policy.model.proprio_encoder.training

    policy.eval()
    assert not policy.training
    assert not policy.model.mot.mixtures["action"].training


def test_policy_replay_contract_is_exact_and_tensor_only():
    policy = FastWAMPolicy(
        model=_TinyFastWAM(),
        processor=object(),
        config=FastWAMPolicyConfig(),
    )
    replay = _policy_replay()
    assert policy._validate_replay(replay) == 1

    wrong_dtype = dict(replay)
    wrong_dtype["chains"] = wrong_dtype["chains"].to(torch.bfloat16)
    with pytest.raises(TypeError, match="chains must have dtype"):
        policy._validate_replay(wrong_dtype)

    wrong_context = dict(replay)
    wrong_context["text_context"] = torch.zeros(1, 128, 2048)
    with pytest.raises(ValueError, match=r"\[B,128,4096\]"):
        policy._validate_replay(wrong_context)

    unexpected = dict(replay)
    unexpected["prompt"] = "not a tensor"
    with pytest.raises(ValueError, match="unexpected"):
        policy._validate_replay(unexpected)


def test_ppo_value_head_uses_model_dtype_but_returns_fp32_and_detaches_features():
    value_head = nn.Sequential(
        nn.Linear(3072, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    ).to(dtype=torch.bfloat16)
    policy = FastWAMPolicy(
        model=_TinyFastWAM(),
        processor=object(),
        config=FastWAMPolicyConfig(detach_critic_input=True),
        value_head=value_head,
    )
    features = torch.randn(2, 3072, dtype=torch.float32, requires_grad=True)

    values = policy._compute_values(features)
    values.sum().backward()

    assert values.shape == (2, 1)
    assert values.dtype == torch.float32
    assert features.grad is None
    assert all(
        parameter.dtype == torch.bfloat16 for parameter in value_head.parameters()
    )
    assert all(parameter.grad is not None for parameter in value_head.parameters())
