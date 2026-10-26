# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Derive an isolated MFD smoke configuration from the accepted DMD-new config."""

import argparse
import copy
import json
from pathlib import Path


def config_from_dmd(base: dict) -> dict:
    """Inherit model/data/batch/optimizer settings; change only the MFD method."""
    required = {
        "method": "onedp",
        "batch_size": 384,
        "microbatch_per_rank": 192,
        "auxiliary_updates_per_student": 5,
        "onedp_loss_action_dim": 14,
        "lr": 1e-5,
        "aux_lr": 2e-4,
    }
    for key, expected in required.items():
        if base.get(key) != expected:
            raise ValueError(f"Expected accepted DMD-new {key}={expected}")
    if not base.get("data_manifest"):
        raise ValueError("MFD baseline requires the accepted multitask manifest")
    result = copy.deepcopy(base)
    result.pop("onedp_loss_action_dim")
    result.update(
        method="mfd",
        mfd_step_size=0.02,
        mfd_loss_action_dim=14,
        log_every=1,
        checkpoint_keep_steps=[12, 14],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = config_from_dmd(json.loads(Path(args.base_config).read_text()))
    # Do not overwrite an active experiment configuration.
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
