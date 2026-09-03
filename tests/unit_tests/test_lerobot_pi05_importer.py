# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json

import pytest
import torch
from safetensors.torch import save_file

from rlinf.utils.ckpt_convertor.openpi import lerobot_pi05_to_openpi_rlinf as importer


def test_source_contract_requires_one_model_wrapper(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"model.weight": torch.zeros(2, 3)}, path)
    assert importer._source_contract(path) == {
        "weight": ((2, 3), "torch.float32")
    }


def test_source_contract_rejects_unwrapped_key(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.zeros(2, 3)}, path)
    with pytest.raises(ValueError, match="model. prefix"):
        importer._source_contract(path)


def test_norm_stats_are_mean_std_and_14d(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    processor = source / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    save_file(
        {
            "observation.state.mean": torch.arange(14, dtype=torch.float32),
            "observation.state.std": torch.ones(14),
            "action.mean": torch.arange(14, dtype=torch.float32) + 1,
            "action.std": torch.ones(14) * 2,
        },
        processor,
    )
    output = tmp_path / "output"
    path = importer._write_norm_stats(source, output)
    payload = json.loads(path.read_text())["norm_stats"]
    assert len(payload["state"]["mean"]) == 14
    assert len(payload["actions"]["mean"]) == 14
    assert payload["actions"]["std"] == [2.0] * 14
