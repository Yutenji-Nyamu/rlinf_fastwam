# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Storage pruning preserves resume/evaluation points and partial saves."""

import json

import pytest

from rlinf.algorithms.pi05_distillation.checkpoint_retention import prune_checkpoints


def test_keep_recent_full_pairs_and_eval_points(tmp_path):
    for step in (10, 20, 30, 40, 50):
        for kind in ("checkpoint", "policy"):
            (tmp_path / f"{kind}-{step:06d}.pt").write_bytes(b"fixture")
    (tmp_path / "checkpoint-000060.pt.partial").write_bytes(b"partial")
    (tmp_path / "checkpoint-000005.pt").write_bytes(b"unpaired")
    receipt = {
        "step": 50,
        "path": str(tmp_path / "checkpoint-000050.pt"),
        "policy": str(tmp_path / "policy-000050.pt"),
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(receipt))
    deleted = prune_checkpoints(tmp_path, 2, [20])
    assert {r["step"] for r in deleted} == {10, 30}
    for step in (20, 40, 50):
        for kind in ("checkpoint", "policy"):
            assert (tmp_path / f"{kind}-{step:06d}.pt").is_file()
    assert (tmp_path / "checkpoint-000005.pt").is_file()
    assert (tmp_path / "checkpoint-000060.pt.partial").is_file()
    assert prune_checkpoints(tmp_path, 2, [20]) == []


def test_no_deletion_without_complete_current_receipt(tmp_path):
    (tmp_path / "checkpoint.json").write_text(json.dumps({"step": 50}))
    (tmp_path / "checkpoint-000050.pt").write_bytes(b"not-published-pair")
    with pytest.raises(ValueError, match="complete local pair"):
        prune_checkpoints(tmp_path, 2, [])
    assert (tmp_path / "checkpoint-000050.pt").exists()


def test_explicit_latest_only_overrides_historical_best(tmp_path):
    for step in (10, 20, 30):
        for kind in ("checkpoint", "policy"):
            (tmp_path / f"{kind}-{step:06d}.pt").write_bytes(b"fixture")
    (tmp_path / "checkpoint.json").write_text(json.dumps({
        "step": 30, "path": str(tmp_path / "checkpoint-000030.pt"),
        "policy": str(tmp_path / "policy-000030.pt"),
    }))
    (tmp_path / "retention-policy.json").write_text(
        json.dumps({"mode": "latest_complete_only"})
    )
    (tmp_path / "checkpoint-000040.pt").write_bytes(b"unfinished")
    deleted = prune_checkpoints(tmp_path, 2, [10])
    assert {row["step"] for row in deleted} == {10, 20}
    assert (tmp_path / "checkpoint-000030.pt").exists()
    assert (tmp_path / "checkpoint-000040.pt").exists()
