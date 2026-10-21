# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Episode-level early-alarm calibration; the holdout never selects a threshold."""

from __future__ import annotations

import math

import numpy as np


def wilson(successes: int, total: int) -> list[float] | None:
    """Return a descriptive 95% Wilson interval for an episode proportion."""
    if not total:
        return None
    p, z = successes / total, 1.959963984540054
    d = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [middle - half, middle + half]


def metrics(rows: list[dict], threshold: float, score_key: str = "probability") -> dict:
    """Count any alarm before native success, including successful episodes."""
    episodes: dict[str, list] = {}
    for row in rows:
        score = float(row[score_key])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Scores must be finite probabilities")
        episodes.setdefault(str(row["actual_seed"]), []).append(row)
    early = eligible = detected = positive = tp = fp = tn = fn = 0
    for values in episodes.values():
        negatives = [r for r in values if not r["native_success_once"]]
        positives = [r for r in values if r["native_success_once"]]
        eligible += bool(negatives)
        early += any(float(r[score_key]) >= threshold for r in negatives)
        positive += bool(positives)
        detected += any(float(r[score_key]) >= threshold for r in positives)
        for r in values:
            hit = float(r[score_key]) >= threshold
            truth = bool(r["native_success_once"])
            tp += hit and truth
            fp += hit and not truth
            tn += not hit and not truth
            fn += not hit and truth
    return {
        "threshold": float(threshold),
        "episodes": len(episodes),
        "early_alarm_episodes": early,
        "episodes_with_negative_prefix": eligible,
        "early_alarm_rate": early / eligible if eligible else None,
        "early_alarm_wilson95": wilson(early, eligible),
        "successful_episodes": positive,
        "detected_successful_episodes": detected,
        "success_recall": detected / positive if positive else None,
        "success_recall_wilson95": wilson(detected, positive),
        "window_counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def select_threshold(rows: list[dict]) -> dict:
    """Choose on calibration only, requiring zero early alarms and recall >= .5."""
    if not rows or any(r["split"] != "calibration" for r in rows):
        raise ValueError("Only calibration rows may select the threshold")
    scores = np.asarray([r["probability"] for r in rows], dtype=np.float32)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid calibration probabilities")
    candidates = np.unique(
        np.concatenate((scores, np.nextafter(scores, np.float32(np.inf)), [0.46, 1.0]))
    )
    valid = []
    for threshold in candidates:
        if not 0 < threshold <= 1:
            continue
        m = metrics(rows, float(threshold))
        if m["early_alarm_episodes"] == 0 and (m["success_recall"] or 0) >= 0.5:
            valid.append(m)
    if not valid:
        return {
            "feasible": False,
            "threshold": None,
            "reason": "No threshold gives zero calibration early alarms and >=50% successful-episode recall",
        }
    chosen = max(valid, key=lambda r: (r["success_recall"], r["threshold"]))
    return dict(feasible=True, **chosen)


def report(rows: list[dict], rm_threshold: float) -> dict:
    """Freeze the selected threshold before producing holdout diagnostics."""
    cal = [r for r in rows if r["split"] == "calibration"]
    held = [r for r in rows if r["split"] == "holdout"]
    if {r["actual_seed"] for r in cal} & {r["actual_seed"] for r in held}:
        raise ValueError("Actual seed leaks across calibration/holdout")
    chosen = select_threshold(cal)
    result = {
        "selection": chosen,
        "official_reference": {
            s: metrics(v, 0.46) for s, v in [("calibration", cal), ("holdout", held)]
        },
        "old_rm": {
            s: metrics(v, rm_threshold, "old_rm_probability")
            for s, v in [("calibration", cal), ("holdout", held)]
        },
        "old_rm_independent_test": metrics(
            [r for r in held if r["legacy_rm_split"] == "test"],
            rm_threshold,
            "old_rm_probability",
        ),
        "admissible": False,
    }
    if chosen["feasible"]:
        m = metrics(held, chosen["threshold"])
        result["holdout"] = m
        result["admissible"] = (
            len({r["actual_seed"] for r in cal}) == 64
            and m["episodes"] == 64
            and m["successful_episodes"] >= 20
            and m["early_alarm_rate"] is not None
            and m["early_alarm_rate"] <= 0.05
            and (m["success_recall"] or 0) >= 0.5
        )
    return result
