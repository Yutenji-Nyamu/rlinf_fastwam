# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Reuse GAD's accepted 12->14/export smoke; no formal work on smoke cards."""

import json
import os
import time
from pathlib import Path

import gad_queue as existing

from rlinf.algorithms.pi05_distillation.ot import validate_config


def main():
    q = existing.q
    q.save(q.D / "smoke-identity.json", q.proc(os.getpid()))
    assert validate_config(q.read(q.D / "endpoint.json"))
    try:
        result = existing.smoke("endpoint")
        output = q.read(q.D / "formal-plan.json")["smoke_output"]
        rows = [
            json.loads(line)
            for line in (Path(output) / "smoke-gad-endpoint/metrics-rank0.jsonl")
            .read_text()
            .splitlines()
        ]
        for row in rows:
            assert row["ot_cost_reduction"] >= -1e-6
            assert 0 <= row["ot_reassigned_fraction"] <= 1
        result.update(
            passed=True,
            formal_started=False,
            transport=rows[-1],
            version="gad-ot-balanced-k4",
        )
        q.save(q.D / "gad-ot-smoke-summary.json", result)
        q.event(
            "GAD_OT_SMOKE_PASSED",
            peak_gib=result["peak_gib"],
            micro=result["microbatch_per_rank"],
        )
    finally:
        existing.base.clear_owned()
        q.save(
            q.D / "smoke-released.json",
            {"time": time.time(), "physical_gpus": list(q.CARDS)},
        )


if __name__ == "__main__":
    main()
