# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Write RLT training overrides anchored to the checkpoint's saved update count."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--saved-updates", required=True, type=int)
p.add_argument("--output", required=True)
a = p.parse_args()
if a.saved_updates < 0:
    p.error("--saved-updates must be nonnegative")
cfg = {
    "algorithm": {
        "q_output_feature_normalization": True,
        "critic_target_min": 0.0,
        "critic_target_max": 1.0,
        "rlt_reference_bootstrap": False,
        "reference_dropout_prob": 0.0,
        "bc_weight": 1.0,
        "q_weight": 0.0,
        "actor_weight_schedule": {
            "enable": True,
            "warmup_updates": a.saved_updates + 2000,
            "ramp_updates": 10000,
            "warmup_bc_weight": 1.0,
            "online_bc_weight": 1.0,
            "warmup_q_weight": 0.0,
            "online_q_weight": 0.05,
        },
    }
}
Path(a.output).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
