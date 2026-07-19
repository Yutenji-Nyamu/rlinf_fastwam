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

"""Unit tests for RLinf Fast-WAM checkpoint export."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from rlinf.models.embodiment.fastwam.export import (
    export_deploy_checkpoint,
    extract_official_fastwam_payload,
    load_rlinf_model_state,
    resolve_actor_checkpoint,
    validate_official_schema,
)


def _base_checkpoint() -> dict:
    return {
        "mot": {
            "mixtures.video.proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
            "mixtures.action.proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        },
        "proprio_encoder": {
            "weight": torch.full((2, 2), 2.0, dtype=torch.bfloat16),
            "bias": torch.full((2,), 3.0, dtype=torch.bfloat16),
        },
        "step": None,
        "torch_dtype": "torch.bfloat16",
    }


def _rlinf_state() -> dict:
    return {
        "model.mot.mixtures.video.proj.weight": torch.full(
            (2, 2), 4.0, dtype=torch.bfloat16
        ),
        "model.mot.mixtures.action.proj.weight": torch.full(
            (2, 2), 5.0, dtype=torch.bfloat16
        ),
        "model.proprio_encoder.weight": torch.full((2, 2), 6.0, dtype=torch.bfloat16),
        "model.proprio_encoder.bias": torch.full((2,), 7.0, dtype=torch.bfloat16),
        # These frozen common components exist in the RLinf full model state
        # but are intentionally loaded separately by official Fast-WAM deploy.
        "model.vae.model.scale": torch.tensor(1.0),
        "model.text_encoder.token_embedding.weight": torch.zeros(1),
    }


def _assert_state_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_extracts_exact_official_payload_and_ignores_common_components():
    base = _base_checkpoint()
    state = _rlinf_state()

    payload = extract_official_fastwam_payload(state, base, step=9)

    assert set(payload) == {
        "mot",
        "proprio_encoder",
        "step",
        "torch_dtype",
    }
    assert payload["step"] == 9
    assert payload["torch_dtype"] == "torch.bfloat16"
    _assert_state_equal(
        payload["mot"],
        {
            "mixtures.video.proj.weight": state["model.mot.mixtures.video.proj.weight"],
            "mixtures.action.proj.weight": state[
                "model.mot.mixtures.action.proj.weight"
            ],
        },
    )
    _assert_state_equal(
        payload["proprio_encoder"],
        {
            "weight": state["model.proprio_encoder.weight"],
            "bias": state["model.proprio_encoder.bias"],
        },
    )


def test_ppo_value_head_stays_in_rlinf_state_but_is_excluded_from_deploy_payload():
    state = _rlinf_state()
    state["value_head.mlp.0.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)

    payload = extract_official_fastwam_payload(state, _base_checkpoint(), step=2)

    assert "value_head" not in payload
    assert set(payload) == {"mot", "proprio_encoder", "step", "torch_dtype"}


@pytest.mark.parametrize(
    "alias_key",
    [
        "model.video_expert.proj.weight",
        "model.action_expert.proj.weight",
        "model.dit.mixtures.action.proj.weight",
    ],
)
def test_rejects_duplicate_module_alias_prefixes(alias_key):
    state = _rlinf_state()
    state[alias_key] = torch.zeros(2, 2, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="Forbidden Fast-WAM alias"):
        extract_official_fastwam_payload(state, _base_checkpoint(), step=1)


def test_rejects_unknown_wrapper_prefix_instead_of_guessing():
    state = _rlinf_state()
    state["module.model.mot.mixtures.action.proj.weight"] = torch.zeros(
        2, 2, dtype=torch.bfloat16
    )

    with pytest.raises(ValueError, match="Unexpected RLinf state-dict keys"):
        extract_official_fastwam_payload(state, _base_checkpoint(), step=1)


def test_rejects_missing_mot_key():
    state = _rlinf_state()
    del state["model.mot.mixtures.video.proj.weight"]

    with pytest.raises(ValueError, match="mot key mismatch"):
        extract_official_fastwam_payload(state, _base_checkpoint(), step=1)


def test_rejects_extra_mot_key():
    state = _rlinf_state()
    state["model.mot.mixtures.action.extra.weight"] = torch.zeros(
        1, dtype=torch.bfloat16
    )

    with pytest.raises(ValueError, match="mot key mismatch"):
        extract_official_fastwam_payload(state, _base_checkpoint(), step=1)


def test_rejects_shape_mismatch():
    state = _rlinf_state()
    state["model.mot.mixtures.action.proj.weight"] = torch.zeros(
        3, 2, dtype=torch.bfloat16
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        extract_official_fastwam_payload(state, _base_checkpoint(), step=1)


def test_rejects_dtype_mismatch():
    state = _rlinf_state()
    state["model.mot.mixtures.action.proj.weight"] = torch.zeros(
        2, 2, dtype=torch.float32
    )

    with pytest.raises(ValueError, match="dtype mismatch"):
        extract_official_fastwam_payload(state, _base_checkpoint(), step=1)


def test_rejects_missing_proprio_encoder():
    state = {
        key: value
        for key, value in _rlinf_state().items()
        if not key.startswith("model.proprio_encoder.")
    }

    with pytest.raises(ValueError, match="has proprio_encoder"):
        extract_official_fastwam_payload(state, _base_checkpoint(), step=1)


def test_schema_rejects_unofficial_top_level_key():
    payload = extract_official_fastwam_payload(
        _rlinf_state(), _base_checkpoint(), step=1
    )
    payload["optimizer"] = {}

    with pytest.raises(ValueError, match="top-level key mismatch"):
        validate_official_schema(payload, _base_checkpoint())


def test_export_rejects_boolean_step(tmp_path):
    full_weights = tmp_path / "full_weights.pt"
    base_path = tmp_path / "base.pt"
    torch.save(_rlinf_state(), full_weights)
    torch.save(_base_checkpoint(), base_path)

    with pytest.raises(ValueError, match="non-negative integer"):
        export_deploy_checkpoint(
            checkpoint_path=full_weights,
            base_checkpoint_path=base_path,
            output_path=tmp_path / "deploy.pt",
            step=True,
        )


def _make_fake_dcp_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / ".metadata").write_bytes(b"metadata")
    (path / "__0_0.distcp").write_bytes(b"shard")


def test_resolves_global_step_and_direct_dcp_paths(tmp_path):
    global_step = tmp_path / "global_step_12"
    dcp_path = global_step / "actor" / "dcp_checkpoint"
    _make_fake_dcp_dir(dcp_path)

    from_global_step = resolve_actor_checkpoint(global_step)
    from_dcp = resolve_actor_checkpoint(dcp_path)

    assert from_global_step.kind == "dcp"
    assert from_global_step.path == dcp_path.resolve()
    assert from_global_step.step == 12
    assert from_dcp.kind == "dcp"
    assert from_dcp.step == 12


def test_rejects_actor_directory_as_ambiguous_input(tmp_path):
    actor = tmp_path / "global_step_2" / "actor"
    _make_fake_dcp_dir(actor / "dcp_checkpoint")

    with pytest.raises(ValueError, match="global_step_N directory"):
        resolve_actor_checkpoint(actor)


def test_full_weights_export_round_trip(tmp_path):
    full_weights = tmp_path / "full_weights.pt"
    base_path = tmp_path / "base.pt"
    output = tmp_path / "deploy" / "fastwam_step_7.pt"
    torch.save(_rlinf_state(), full_weights)
    torch.save(_base_checkpoint(), base_path)

    exported = export_deploy_checkpoint(
        checkpoint_path=full_weights,
        base_checkpoint_path=base_path,
        output_path=output,
        step=7,
    )

    assert exported == output.resolve()
    payload = torch.load(exported, map_location="cpu", weights_only=True)
    assert payload["step"] == 7
    validate_official_schema(payload, _base_checkpoint())
    assert not list(output.parent.glob(".*.tmp"))


def test_export_refuses_existing_output_without_overwrite(tmp_path):
    full_weights = tmp_path / "full_weights.pt"
    base_path = tmp_path / "base.pt"
    output = tmp_path / "deploy.pt"
    torch.save(_rlinf_state(), full_weights)
    torch.save(_base_checkpoint(), base_path)
    output.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="already exists"):
        export_deploy_checkpoint(
            checkpoint_path=full_weights,
            base_checkpoint_path=base_path,
            output_path=output,
            step=1,
        )
    assert output.read_bytes() == b"keep"


def _dcp_private_api_available() -> bool:
    return (
        importlib.util.find_spec("torch.distributed.checkpoint.format_utils")
        is not None
        and importlib.util.find_spec("torch.distributed.checkpoint.state_dict_loader")
        is not None
    )


@pytest.mark.skipif(
    not _dcp_private_api_available(),
    reason="locked Torch 2.7.1 private DCP API is unavailable",
)
def test_synthetic_dcp_no_dist_model_state_round_trip(tmp_path):
    from torch.distributed import checkpoint as dcp

    dcp_path = tmp_path / "global_step_3" / "actor" / "dcp_checkpoint"
    state = _rlinf_state()
    dcp.save(
        {"fsdp_checkpoint": {"model": state}},
        checkpoint_id=str(dcp_path),
    )

    loaded = load_rlinf_model_state(dcp_path)

    _assert_state_equal(dict(loaded), state)
