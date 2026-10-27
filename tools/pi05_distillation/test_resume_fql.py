# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""An interrupted FQL handoff retains results from earlier resumed sweeps."""

import importlib.util
import json
import sys
import types
from pathlib import Path


def test_resume_uses_original_and_continued_task_results(tmp_path, monkeypatch):
    directory, output = tmp_path / "queue", tmp_path / "output"
    output.mkdir()

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def row(task, successes):
        return {"task": task, "valid": True, "episodes": 8, "successes": successes}

    save(directory / "task-list.json", {"tasks": [{"task": t} for t in "abc"]})
    save(directory / "eval-fql-s001130/progress.json", {"results": [row("a", 2)]})
    save(
        directory / "eval-fql-s001130-continued-100/progress.json",
        {"results": [row("b", 3)]},
    )
    calls = []

    def sweep(*args, tasks, **kwargs):
        calls.append(tasks)
        return {"tasks": [row("c", 4)]}

    monkeypatch.setitem(
        sys.modules,
        "run_queue",
        types.SimpleNamespace(
            D=directory,
            O=output,
            read=lambda path: json.loads(path.read_text()),
            save=save,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "multitask_eval", types.SimpleNamespace(sweep=sweep)
    )
    monkeypatch.setattr(sys, "argv", ["resume", str(directory), str(tmp_path)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = Path(__file__).with_name("resume_existing_fql.py")
    spec = importlib.util.spec_from_file_location("resume_fql_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.resume_evaluation({"step": 1130, "policy": "existing-policy.pt"})
    assert calls == [[{"task": "c"}]]
    assert result["resumed_from_completed_tasks"] == 2
    assert result["episodes"] == 24 and result["successes"] == 9
    module.resume_evaluation({"step": 1130, "policy": "existing-policy.pt"})
    assert len(calls) == 1
