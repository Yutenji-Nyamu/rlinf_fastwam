from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path("audits/20260718-1932-move-stapler-deep-audit")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_./]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def parse_metrics(path: Path) -> list[dict[str, float]]:
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    starts = list(re.finditer(r"Global Step:\s*(\d+)/100", text))
    rows: list[dict[str, float]] = []
    for i, match in enumerate(starts):
        block = text[match.start() : starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        row: dict[str, float] = {"global_step": float(match.group(1))}
        step_time = re.search(r"Step Time:\s*([0-9.]+)s", block)
        if step_time:
            row["step_time"] = float(step_time.group(1))
        for key, val in PAIR.findall(block):
            row[key] = float(val)
        rows.append(row)
    return rows


def mean(rows: list[dict[str, float]], key: str) -> float:
    return statistics.fmean(r[key] for r in rows if key in r)


def lin_slope(rows: list[dict[str, float]], key: str) -> float:
    xs = [r["global_step"] for r in rows if key in r]
    ys = [r[key] for r in rows if key in r]
    xbar, ybar = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)


rows = parse_metrics(ROOT / "metrics.log")
for row in rows:
    row["success_count"] = round(row["success_once"] * row["num_trajectories"])

resource_rows: list[dict[str, str]] = []
with (ROOT / "resources.csv").open(newline="", encoding="utf-8") as handle:
    resource_rows = list(csv.DictReader(handle))

numeric_resource: dict[str, list[float]] = {}
for key in resource_rows[0]:
    vals: list[float] = []
    for row in resource_rows:
        try:
            vals.append(float(row[key]))
        except (TypeError, ValueError):
            pass
    if vals:
        numeric_resource[key] = vals

summary = {
    "n_steps": len(rows),
    "first_step": int(rows[0]["global_step"]),
    "last_step": int(rows[-1]["global_step"]),
    "success": {
        "all_mean": mean(rows, "success_once"),
        "first5": mean(rows[:5], "success_once"),
        "last5": mean(rows[-5:], "success_once"),
        "first10": mean(rows[:10], "success_once"),
        "middle10_steps_10_19": mean(rows[9:19], "success_once"),
        "last10": mean(rows[-10:], "success_once"),
        "slope_per_step": lin_slope(rows, "success_once"),
        "max": max((r["success_once"], int(r["global_step"])) for r in rows),
        "min": min((r["success_once"], int(r["global_step"])) for r in rows),
        "counts": [(int(r["global_step"]), int(r["success_count"])) for r in rows],
    },
    "optimizer": {
        key: {
            "mean": mean(rows, key),
            "min": min(r[key] for r in rows),
            "max": max(r[key] for r in rows),
            "latest": rows[-1][key],
        }
        for key in [
            "actor/approx_kl",
            "actor/clip_fraction",
            "actor/ratio",
            "actor/ratio_abs",
            "actor/grad_norm",
            "actor/policy_loss",
            "actor/policy_loss_abs",
            "actor/total_loss",
            "advantages_mean",
            "advantages_min",
            "advantages_max",
            "step_time",
            "actor/run_training",
            "rollout/predict",
            "env/interact",
        ]
    },
    "nonfinite_metric_values": [
        (int(row["global_step"]), key, val)
        for row in rows
        for key, val in row.items()
        if not math.isfinite(val)
    ],
    "resources": {
        key: {"min": min(vals), "max": max(vals), "last": vals[-1], "mean": statistics.fmean(vals)}
        for key, vals in numeric_resource.items()
        if any(token in key for token in ["ram", "gpu", "rss", "shm"])
    },
}

print(json.dumps(summary, indent=2, ensure_ascii=False))
