# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Index completed transition episodes without copying success/failure payloads."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from rlinf.algorithms.pi05_distillation.transitions import build_manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    a = p.parse_args()
    m = build_manifest(a.roots, a.output)
    if a.config:
        path = Path(a.config)
        c = json.loads(path.read_text())
        count = sum(r["transitions"] for r in m["episodes"])
        per_epoch = math.ceil(count / c["batch_size"])
        total = c["epochs"] * per_epoch
        c.update(
            transition_manifest=str(Path(a.output).resolve()),
            budget_resolved=True,
            checkpoint_steps=max(1, total // 10),
            warmup_steps=min(167, max(1, total // 20)),
            checkpoint_keep_steps=sorted(
                {
                    max(1, round(total * f))
                    for f in [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1]
                }
            ),
        )
        path.write_text(json.dumps(c, indent=2))
    print(
        json.dumps(
            {
                "episodes": len(m["episodes"]),
                "successes": sum(r["success"] for r in m["episodes"]),
                "transitions": sum(r["transitions"] for r in m["episodes"]),
                "per_task": dict(Counter(r["task"] for r in m["episodes"])),
            }
        )
    )
