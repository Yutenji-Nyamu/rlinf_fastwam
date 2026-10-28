# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Three isolated v2 methods: accepted pair launcher, 12->14, native N1, STOP."""

import os
import sys
import traceback

import run_queue as q

from rlinf.algorithms.pi05_distillation.variants import V2_METHODS, resolve_method


def main():
    q.save(q.D / "smoke-identity.json", q.proc(os.getpid()))
    flags = ["--decode-workers", "8", "--prefetch-batches", "1", "--profile-stages"]
    selected = sys.argv[2:] or list(V2_METHODS)
    assert all(name in V2_METHODS for name in selected)
    summary = q.D / "v2-smoke-summary.json"
    results = q.read(summary) if summary.exists() else {}
    for variant in selected:
        config = q.read(q.D / f"{variant}.json")
        method, dim = resolve_method(config)
        assert method == V2_METHODS[variant] and dim == 14
        output = q.O / f"smoke-{variant}"
        try:
            first = q.train(
                variant, output, 12, label=f"{variant}-12", extra_args=flags
            )
            q.train(
                variant,
                output,
                14,
                first["path"],
                label=f"{variant}-resume-14",
                extra_args=flags,
            )
            q.accept_smoke(variant)
            resolved = q.read(output / "resolved.json")
            assert (
                resolved["variant"] == variant
                and resolved["student_loss_action_dim"] == 14
            )
            if method in {"onedp", "mfd"}:
                assert resolved["auxiliary_loss_action_dim"] == 14
                for check in q.read(q.D / f"smoke-{variant}-accepted.json")["checks"]:
                    row = check["last_update"]
                    assert (
                        row["student_updates"] == 14 and row["auxiliary_updates"] == 70
                    )
            results[variant] = {"passed": True, "output": str(output)}
            q.event("PHYSICAL_V2_METHOD_PASSED", variant=variant)
        except Exception:
            results[variant] = {"passed": False, "error": traceback.format_exc()}
            q.event(
                "PHYSICAL_V2_METHOD_FAILED",
                variant=variant,
                error=results[variant]["error"],
            )
        q.save(summary, results)
    passed = all(row["passed"] for row in results.values())
    q.event("PHYSICAL_V2_SMOKE_COMPLETE", passed=passed, formal_started=False)
    if not passed:
        raise RuntimeError("One or more v2 smoke stages failed; see summary")


if __name__ == "__main__":
    main()
