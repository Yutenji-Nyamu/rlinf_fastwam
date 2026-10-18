# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Freeze high/low threshold sweeps from independent weak-policy calibration."""

import copy
import shutil
from pathlib import Path

import numpy as np
from protocol import read, save, validate_config

# Cover the range early without choosing the order from evaluation outcomes.
FRACTIONS = (0.10, 0.90, 0.30, 0.70, 0.50, 0.05, 0.95, 0.20, 0.80, 0.40, 0.60, 0.15, 0.85, 0.25, 0.75)


def conditions(calibration):
    """Return 30 frozen gates; target fractions refer only to calibration."""
    values = np.asarray([row["u"] for row in calibration], dtype=float)
    assert len(values) and np.isfinite(values).all()
    rows = []
    for fraction in FRACTIONS:
        for policy in ("high", "low"):
            quantile = 1 - fraction if policy == "high" else fraction
            threshold = float(np.quantile(values, quantile))
            selected = values > threshold if policy == "high" else values < threshold
            rows.append({
                "condition": f"cp180_{policy}_p{round(100 * fraction):02d}",
                "policy": policy,
                "target_fraction": fraction,
                "calibration_quantile": quantile,
                "threshold": threshold,
                "calibration_fraction": float(selected.mean()),
            })
    return rows


def prepare(old_root, root, transaction, calibration_path, source_commit):
    """Reuse accepted seeds/configs/video; change only each condition's gate."""
    old_root, root, transaction = map(Path, (old_root, root, transaction))
    calibration_path = Path(calibration_path)
    rows = conditions(read(calibration_path))
    assert not root.exists()
    for directory in ("configs", "thresholds", "batches", "summaries"):
        (root / directory).mkdir(parents=True)
    for name in ("seeds.json", "threshold.json"):
        shutil.copy2(old_root / name, root / name)
    shutil.copy2(calibration_path, root / "calibration-decisions.json")
    for source in (old_root / "summaries").glob("*.json"):
        value = read(source)
        if not source.name.endswith("-episodes.json"):
            value["reused_from"] = str(source)
        save(root / "summaries" / source.name, value)
    save(root / "summary.json", [
        read(path) for path in sorted((root / "summaries").glob("*.json"))
        if not path.name.endswith("-episodes.json")
    ])
    plans = {gpu: {"gpu": gpu, "stages": []} for gpu in (6, 7)}
    configs = []
    for row in rows:
        name = row["condition"]
        threshold_path = root / "thresholds" / (name + ".json")
        gate = read(old_root / "threshold.json")
        gate.update({row["policy"]: row["threshold"], **row})
        save(threshold_path, gate)
        for gpu in (6, 7):
            stage = {"condition": name, "configs": []}
            for batch in (gpu - 6, gpu - 4):
                config = copy.deepcopy(read(old_root / "configs" / f"cp180_high_u_b{batch}.json"))
                out = root / "batches" / f"{name}_b{batch}"
                seeds = root / "configs" / f"seeds_b{batch}.json"
                if not seeds.exists():
                    shutil.copy2(config["env"]["seeds_path"], seeds)
                config.update(condition=name, policy=row["policy"], output=str(out),
                              source_commit=source_commit, threshold_path=str(threshold_path),
                              sweep_target_fraction=row["target_fraction"])
                config["env"]["seeds_path"] = str(seeds)
                config["env"]["task_config"]["save_path"] = str(out / "env")
                config["env"]["video_cfg"]["video_base_dir"] = str(out / "unused")
                validate_config(config)
                assert config["gpu"] == gpu and config["batch"] == batch
                assert config["record_full_video"] and config["takeover_mode"] == "weights"
                path = root / "configs" / f"{name}_b{batch}.json"
                save(path, config)
                configs.append(str(path))
                stage["configs"].append(str(path))
            plans[gpu]["stages"].append(stage)
    for gpu, plan in plans.items():
        save(transaction / f"lane-g{gpu}-plan.json", plan)
    manifest = {"source_commit": source_commit, "calibration_decisions": len(read(calibration_path)),
                "conditions": rows, "configs": configs, "episodes_per_condition": 128,
                "total_new_episodes": 3840, "per_card_parallelism": 32, "physical_gpus": [6, 7],
                "expected_wall_hours": [22, 24], "reused_from": str(old_root)}
    save(root / "sweep-manifest.json", manifest)
    return manifest
