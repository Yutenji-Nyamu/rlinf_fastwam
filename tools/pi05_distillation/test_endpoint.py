# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Endpoint gradient isolation and the distinct CP/endpoint inference interface."""

import importlib.machinery
import importlib.util
import json
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
        sys.modules,
        "multitask_eval",
        types.SimpleNamespace(sweep=None, completed_checkpoint=None),
    )
    with pytest.raises(SystemExit, match="143"):
        runpy.run_path(
            str(Path(__file__).with_name("endpoint_queue.py")), run_name="__main__"
        )
    assert "identity.json" in saved
    assert "released.json" not in saved


def test_onedp_multitask_keeps_method_budget_and_routes_evaluations(
    tmp_path, monkeypatch
):
    directory, output, data = (tmp_path / n for n in ("queue", "output", "data"))
    for p in (directory, output, data):
        p.mkdir()

    def save(path, value):
        path.write_text(json.dumps(value))

    save(
        directory / "prepared.json", {"method": "onedp", "eval_budget_epochs": [1, 20]}
    )
    save(directory / "registered.json", {"gpus": [6, 7], "identity": {"pid": 123}})
    save(
        directory / "schedule.json",
        {
            "start_after_rlt_handoff": True,
            "run_smoke_before_formal": True,
            "data_source": str(data),
            "baseline_source": str(output / "pending"),
        },
    )
    save(
        directory / "onedp.json",
        {
            "epochs": 20,
            "batch_size": 128,
            "microbatch_per_rank": 64,
            "checkpoint_keep_last": 2,
        },
    )
    save(data / "data-ready.json", {"tasks": 50, "episodes": 2500, "windows": 427287})
    save(data / "data-manifest.json", {})
    trains, evaluations = [], []

    def train(method, out, stop, resume=None, label=None):
        trains.append((method, stop, resume))
        return {
            "step": stop,
            "path": str(out / f"cp{stop}"),
            "policy": str(out / f"p{stop}"),
        }

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.D, module.O, module.CARDS = directory, output, [6, 7]
            module.owned_children = []
            module.proc = lambda pid: {"pid": 123}
            module.save, module.read = save, lambda p: json.loads(p.read_text())
            module.event = lambda *args, **kwargs: None
            module.gpus = lambda: {g: {"processes": []} for g in [6, 7]}
            module.train, module.accept_smoke = train, lambda method: None

    real_spec = importlib.util.spec_from_file_location

    def spec(name, path, *args, **kwargs):
        if Path(path).name == "run_queue.py":
            return importlib.machinery.ModuleSpec(name, Loader())
        return real_spec(name, path, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", spec)
    monkeypatch.setitem(
        sys.modules,
        "multitask_eval",
        types.SimpleNamespace(
            sweep=lambda *args, **kwargs: evaluations.append(kwargs),
            completed_checkpoint=None,
        ),
    )
    runpy.run_path(
        str(Path(__file__).with_name("endpoint_queue.py")), run_name="__main__"
    )
    assert [(method, step) for method, step, _ in trains] == [
        ("onedp", 12),
        ("onedp", 14),
        ("onedp", 3339),
        ("onedp", 66780),
    ]
    assert trains[2][2] is None  # Formal training starts from the original base.
    assert all(e["method"] == "onedp" for e in evaluations)
    assert json.loads((directory / "formal-ready.json").read_text())["epochs"] == 20
    assert json.loads((directory / "onedp.json").read_text())[
        "checkpoint_keep_steps"
    ] == [3339, 66780]
