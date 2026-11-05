# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Rejected optional acceleration returns to the accepted production baseline."""

import json
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def restore_failed_phase(directory: Any, phase: str, code: int) -> None:
    """Disable failed optional switches while retaining accepted settings."""
    d = Path(directory)
    c = OmegaConf.load(d / "config/formal.yaml")
    if phase == "sac_v2":
        c.actor.dsrl_detach_actor_q = False
        c.actor.dsrl_validate_u_on_cpu = False
        c.actor.dsrl_cpu_threads = 0
        c.actor.dsrl_cpu_affinity = []
    else:
        c.env.train["dsrl_acceleration"] = {}
        c.env.eval["dsrl_acceleration"] = {}
        plan = json.loads((d / "plan.json").read_text())
        for p in [d / f"native-g{plan['gpu']}.json"]:
            if not p.exists():
                continue
            native = json.loads(p.read_text())
            native["acceleration"] = {}
            temp = p.with_suffix(".tmp")
            temp.write_text(json.dumps(native, indent=2))
            temp.replace(p)
    OmegaConf.save(c, d / "config/formal.yaml")
    (d / (phase + "-rejected.json")).write_text(
        json.dumps(
            {
                "time": time.time(),
                "exit_code": code,
                "fallback": "existing accepted acceleration only",
            },
            indent=2,
        )
    )


def main() -> None:
    """Dispatch a slot-scoped owner with optional-stage fallback."""
    import os
    import sys

    d = Path(sys.argv[1])
    p = json.loads((d / "plan.json").read_text())
    assert int(p["gpu"]) in (6, 7) and os.getuid() == 1003
    if p["bench"] == "dojo":
        from tools.dsrl_accel_v2.dojo_owner import main as owner
    else:
        from tools.dsrl_accel_v2.rt2_owner import main as owner
    owner(str(d))


if __name__ == "__main__":
    main()
