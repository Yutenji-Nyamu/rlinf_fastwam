# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Budget the measured native single-task adapters, without changing losses."""

import copy
import math

METHODS = ("endpoint", "cp", "onedp", "mfd", "fql_fixed_teacher")
MICRO = {"endpoint": 224, "cp": 80, "onedp": 192, "mfd": 192, "fql_fixed_teacher": 224}
FRACTIONS = (0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)


def evaluation_steps(total):
    """Dense early observations, followed by a bounded twelve-point curve."""
    return sorted({max(1, round(total * f)) for f in FRACTIONS})


def formal_config(reference, windows=4738, budget_epochs=100, transitions=None):
    """Preserve accepted method settings, resolve formal exposure and retention.

    FQL uses real transitions, so its serial budget is tied to the direct
    student's update budget rather than blindly assigning 100 tiny replay epochs.
    """
    c = copy.deepcopy(reference)
    method = c["method"]
    micro = MICRO[method]
    c.update(batch_size=2 * micro, microbatch_per_rank=micro, log_every=10)
    if method == "fql_fixed_teacher":
        if transitions is None:
            c["budget_resolved"] = False
            return c
        if transitions <= 0:
            raise ValueError("FQL requires complete real transitions")
        per_epoch = math.ceil(transitions / c["batch_size"])
        target = budget_epochs * math.ceil(windows / (2 * MICRO["endpoint"]))
        c["epochs"] = math.ceil(target / per_epoch)
        c["budget_resolved"] = True
    else:
        per_epoch = math.ceil(windows / c["batch_size"])
        c["epochs"] = budget_epochs
    total = c["epochs"] * per_epoch
    c.update(
        checkpoint_steps=total + 1,
        checkpoint_keep_last=2,
        checkpoint_keep_steps=evaluation_steps(total),
    )
    if "warmup_steps" in c:
        c["warmup_steps"] = min(500 if method == "cp" else 167, max(1, total // 20))
    return c


def batch_indices(card_index):
    """Two resident waves per card cover the exact 128 paired held-out seeds."""
    if card_index not in (0, 1):
        raise ValueError("Exactly two measured cards are supported")
    return [card_index, card_index + 2]


def run_stages(methods, run, failed):
    """Scientific stage errors advance; cancellation still stops the owner."""
    outcomes = {}
    for method in methods:
        try:
            outcomes[method] = {"status": "complete", "result": run(method)}
        except Exception as exc:
            failed(method, exc)
            outcomes[method] = {"status": "failed", "error": repr(exc)}
    return outcomes
