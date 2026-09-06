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

"""High-information wiring checks for the Fast-WAM RLinf integration."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf


def test_public_model_registry_dispatches_fastwam_lazily(monkeypatch):
    """The public registry must reach Fast-WAM without importing it eagerly."""

    import rlinf.models as model_registry
    import rlinf.models.embodiment.fastwam as fastwam_package

    sentinel = torch.nn.Linear(1, 1)
    calls = []

    def fake_get_model(cfg, torch_dtype):
        calls.append((cfg.model_type, torch_dtype))
        return sentinel

    monkeypatch.setattr(fastwam_package, "get_model", fake_get_model)
    monkeypatch.setattr(model_registry.Worker, "torch_platform", None)
    cfg = OmegaConf.create(
        {
            "model_type": "fastwam_robotwin",
            "precision": "bf16",
            "is_lora": False,
        }
    )

    assert model_registry.get_model(cfg) is sentinel
    assert calls == [("fastwam_robotwin", torch.bfloat16)]


def test_rollout_worker_passes_train_and_eval_mode_via_capability_flag():
    """Fast-WAM opts into mode dispatch without changing other model branches."""

    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    class StubPolicy:
        rlinf_accepts_rollout_mode = True

        def __init__(self):
            self.calls = []

        def predict_action_batch(self, *, env_obs, **kwargs):
            self.calls.append((env_obs, kwargs))
            return torch.zeros(2, 24, 14), {}

    worker = object.__new__(MultiStepRolloutWorker)
    worker._train_sampling_params = {"unused_sampling_knob": 3}
    worker._eval_sampling_params = {"unused_sampling_knob": 5}
    worker.algorithm_cfg = {"loss_type": "actor", "dagger": {}}
    worker.model_cfg = SimpleNamespace(model_type="fastwam_robotwin")
    worker.hf_model = StubPolicy()
    worker.expert_model = None

    predict = MultiStepRolloutWorker.predict
    while hasattr(predict, "__wrapped__"):
        predict = predict.__wrapped__
    train_actions, train_result = predict(worker, {"batch": "train"}, mode="train")
    eval_actions, eval_result = predict(worker, {"batch": "eval"}, mode="eval")

    assert train_actions.shape == (2, 24, 14)
    assert eval_actions.shape == (2, 24, 14)
    assert train_result["expert_label_flag"] is False
    assert eval_result["expert_label_flag"] is False
    assert worker.hf_model.calls == [
        (
            {"batch": "train"},
            {"unused_sampling_knob": 3, "mode": "train"},
        ),
        (
            {"batch": "eval"},
            {"unused_sampling_knob": 5, "mode": "eval"},
        ),
    ]


def test_fsdp_forward_cast_is_opt_in_compatible(monkeypatch):
    """Existing policies keep BF16 input casting; Fast-WAM can disable it."""

    import rlinf.hybrid_engines.fsdp.strategy.fsdp2 as fsdp2_module

    policies = []

    def fake_mixed_precision_policy(**kwargs):
        policies.append(kwargs)
        return kwargs

    def fake_apply_fsdp2_to_model(**kwargs):
        return kwargs["module"]

    monkeypatch.setattr(
        fsdp2_module, "MixedPrecisionPolicy", fake_mixed_precision_policy
    )
    monkeypatch.setattr(
        fsdp2_module, "apply_fsdp2_to_model", fake_apply_fsdp2_to_model
    )

    for explicit_value in (None, False):
        fsdp_config = {
            "mixed_precision": {
                "param_dtype": "bf16",
                "reduce_dtype": "bf16",
            },
            "offload_pin_memory": False,
            "cpu_offload": False,
            "reshard_after_forward": True,
        }
        if explicit_value is not None:
            fsdp_config["cast_forward_inputs"] = explicit_value
        strategy = object.__new__(fsdp2_module.FSDP2Strategy)
        strategy.cfg = OmegaConf.create({"fsdp_config": fsdp_config})
        model = torch.nn.Linear(1, 1)
        assert strategy.wrap_model(model, device_mesh=object()) is model

    assert [policy["cast_forward_inputs"] for policy in policies] == [True, False]
