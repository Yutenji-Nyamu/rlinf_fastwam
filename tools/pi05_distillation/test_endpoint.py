# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Endpoint gradient isolation and the distinct CP/endpoint inference interface."""

import importlib.machinery
import importlib.util
import runpy
import sys
import time
import types
from pathlib import Path

import pytest
import torch

from rlinf.algorithms.pi05_distillation.losses import endpoint_loss


def test_endpoint_same_noise_teacher_frozen_and_padding_excluded():
    class Teacher:
        def __init__(self):
            self.offset = torch.tensor(0.2, requires_grad=True)

        def endpoint(self, noise, context, steps):
            self.noise, self.steps = noise, steps
            return noise + self.offset

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v = torch.nn.Parameter(torch.full((32,), 0.1))

        def forward(self, noise, time, context):
            self.noise, self.time = noise, time
            return self.v.expand_as(noise)

    teacher, student = Teacher(), Student()
    noise = torch.randn(2, 50, 32)
    loss, _ = endpoint_loss(student, teacher, noise, None)
    loss.backward()
    assert teacher.noise is student.noise is noise and teacher.steps == 10
    assert torch.all(student.time == 1) and teacher.offset.grad is None
    torch.testing.assert_close(loss, torch.tensor(0.09))
    torch.testing.assert_close(student.v.grad[:14], torch.full((14,), 0.6 / 14))
    assert torch.count_nonzero(student.v.grad[14:]) == 0


def test_endpoint_evaluation_does_not_create_cp_stop_projection():
    path = Path(__file__).with_name("multitask_eval.py")
    spec = importlib.util.spec_from_file_location("endpoint_evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    template = {"model": {"openpi": {}}, "env": {"task_config": {}, "video_cfg": {}}}
    for method, expected in [("cp", True), ("endpoint", False), ("onedp", False)]:
        config = evaluator.make_config(
            template,
            {"task": "adjust_bottle", "step_limit": 400},
            3,
            Path("/tmp/result"),
            Path("/tmp/seeds.json"),
            "method-test",
            policy="/tmp/policy.pt",
            method=method,
        )
        assert config["model"]["openpi"]["distill_target_time"] is expected
        assert config["distill_weights"] == "/tmp/policy.pt"
    assert template["model"]["openpi"] == {}


def test_cancelling_waiting_successor_never_releases_predecessor(tmp_path, monkeypatch):
    """A temporary GPU-free gap is not ownership of an unfinished predecessor."""
    directory = tmp_path / "queue"
    directory.mkdir()
    (directory / "registered.json").write_text("{}")
    saved = {}

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.D, module.O, module.CARDS = directory, tmp_path / "out", [6, 7]
            module.owned_children = []
            module.proc = lambda pid: {"pid": pid}
            module.save = lambda path, value: saved.update({path.name: value})
            module.event = lambda *args, **kwargs: None
            module.read = lambda path: {
                "predecessor": str(tmp_path / "unfinished-onedp"),
                "predecessor_identity": {"pid": 123},
            }
            module.same = lambda identity: True
            module.gpus = lambda: {g: {"processes": []} for g in [6, 7]}

    real_spec = importlib.util.spec_from_file_location

    def spec(name, path, *args, **kwargs):
        if Path(path).name == "run_queue.py":
            return importlib.machinery.ModuleSpec(name, Loader())
        return real_spec(name, path, *args, **kwargs)

    def cancel(_):
        raise SystemExit(143)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", spec)
    monkeypatch.setattr(time, "sleep", cancel)
    monkeypatch.setitem(
        sys.modules, "multitask_eval", types.SimpleNamespace(sweep=None)
    )
    with pytest.raises(SystemExit, match="143"):
        runpy.run_path(
            str(Path(__file__).with_name("endpoint_queue.py")), run_name="__main__"
        )
    assert "identity.json" in saved
    assert "released.json" not in saved
