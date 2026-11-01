# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""A fresh FQL v2 control from the accepted native FQL-fix configuration."""

import copy
import math

from single_formal_plan import fql_fix_formal_config


def formal_config(reference, transitions, windows=4738):
    if reference.get("experiment_name") != "fql-fix" or reference.get("alpha") != 30:
        raise ValueError("Use the accepted alpha30 FQL-fix control")
    c = fql_fix_formal_config(copy.deepcopy(reference), transitions, windows)
    # The old actual run used 55; do not inherit a recomputed generic warmup.
    c.update(
        warmup_steps=reference["warmup_steps"],
        experiment_name="fql-v2",
        q_agg="min",
        normalize_q_loss=True,
        q_norm_eps=1e-6,
        checkpoint_keep_last=1,
        checkpoint_keep_steps=[],
        retention_mode="latest_complete_only",
        budget_epochs=200,
        updates_per_budget_epoch=math.ceil(windows / c["batch_size"]),
    )
    return c
