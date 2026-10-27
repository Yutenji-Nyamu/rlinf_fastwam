# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Protocol and conditional budget checks; no simulator or GPU required."""

import ast
from pathlib import Path


def test_conditional_boundary():
    path = Path(__file__).with_name("single_teacher_eval.py")
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "needs_short_chunk"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    decide = namespace["needs_short_chunk"]
    baseline = {"episodes": 128, "mean_success_rate": 100 / 128}
    assert decide(baseline, {"episodes": 128, "mean_success_rate": 88 / 128})
    assert not decide(baseline, {"episodes": 128, "mean_success_rate": 87 / 128})
    assert not decide(baseline, {"episodes": 96, "mean_success_rate": 100 / 128})


if __name__ == "__main__":
    test_conditional_boundary()
    print("single-teacher conditional budget check passed")
