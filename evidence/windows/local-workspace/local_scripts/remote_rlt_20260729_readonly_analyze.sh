#!/usr/bin/env bash
set -euo pipefail

/root/autodl-tmp/RLinf/.venv/bin/python -B - <<'PY'
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

runtime = Path(
    "/root/autodl-tmp/experiment_exports/"
    "rlt_stage1_formal_20260729_v1/runtime"
)
run_dir = Path(
    "/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/"
    "robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1"
)
driver_path = runtime / "driver.log"
resource_path = runtime / "resources.csv"
checkpoint_path = run_dir / "checkpoints" / "global_step_2000"


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
driver_text = ansi.sub("", driver_path.read_text(encoding="utf-8", errors="replace"))
metric_by_step: dict[int, dict[str, float]] = {}

for segment in re.split(r"[\r\n]+", driver_text):
    if "Global Step:" not in segment or "train/loss=" not in segment:
        continue
    step_match = re.search(r"(\d+)/2000", segment)
    if not step_match:
        continue
    fields: dict[str, float] = {}
    for field in (
        "time/step",
        "time/training",
        "train/grad_norm",
        "train/learning_rate",
        "train/loss",
        "train/rlt_loss",
        "train/vla_loss",
    ):
        match = re.search(
            re.escape(field) + r"=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
            segment,
            flags=re.IGNORECASE,
        )
        if match:
            fields[field] = float(match.group(1))
    if len(fields) == 7:
        metric_by_step[int(step_match.group(1))] = fields

if set(metric_by_step) != set(range(1, 2001)):
    missing = sorted(set(range(1, 2001)) - set(metric_by_step))
    raise RuntimeError(
        f"metric coverage mismatch: parsed={len(metric_by_step)} "
        f"missing_head={missing[:20]}"
    )

steps = sorted(metric_by_step)
losses = [metric_by_step[step]["train/loss"] for step in steps]
rlt_losses = [metric_by_step[step]["train/rlt_loss"] for step in steps]
vla_losses = [metric_by_step[step]["train/vla_loss"] for step in steps]
grads = [metric_by_step[step]["train/grad_norm"] for step in steps]
lrs = [metric_by_step[step]["train/learning_rate"] for step in steps]
train_times = [metric_by_step[step]["time/training"] for step in steps]
step_times = [metric_by_step[step]["time/step"] for step in steps]

if not all(math.isfinite(value) for values in (losses, grads, lrs, train_times) for value in values):
    raise RuntimeError("non-finite metric")

key_steps = [1, 2, 10, 20, 50, 100, 200, 500, 1000, 1500, 1900, 2000]
key_metrics = []
for step in key_steps:
    fields = metric_by_step[step]
    key_metrics.append(
        {
            "step": step,
            "loss": rounded(fields["train/loss"]),
            "grad_norm": rounded(fields["train/grad_norm"]),
            "lr": fields["train/learning_rate"],
            "training_s": rounded(fields["time/training"]),
            "step_s": rounded(fields["time/step"]),
        }
    )

loss_windows = []
for start in range(1, 2001, 100):
    end = min(start + 99, 2000)
    values = [metric_by_step[step]["train/loss"] for step in range(start, end + 1)]
    loss_windows.append(
        {
            "start": start,
            "end": end,
            "center": (start + end) / 2,
            "mean": rounded(mean(values)),
            "median": rounded(median(values)),
            "p10": rounded(percentile(values, 0.10)),
            "p90": rounded(percentile(values, 0.90)),
        }
    )

plot_training = []
for start in range(1, 2001, 20):
    end = min(start + 19, 2000)
    entries = [metric_by_step[step] for step in range(start, end + 1)]
    plot_training.append(
        {
            "step": end,
            "loss": rounded(mean([entry["train/loss"] for entry in entries])),
            "grad": rounded(mean([entry["train/grad_norm"] for entry in entries])),
            "lr": mean([entry["train/learning_rate"] for entry in entries]),
            "training_s": rounded(
                median([entry["time/training"] for entry in entries])
            ),
        }
    )

with resource_path.open(newline="", encoding="utf-8") as handle:
    resources = [
        {key: float(value) for key, value in row.items()}
        for row in csv.DictReader(handle)
    ]

if not resources:
    raise RuntimeError("empty resources.csv")

resource_start = resources[0]["unix_time"]
active_resources = [
    row for row in resources if row["compute_process_count"] >= 2
]
plot_resource_bins: dict[int, list[dict[str, float]]] = defaultdict(list)
for row in resources:
    bucket = int((row["unix_time"] - resource_start) // 30)
    plot_resource_bins[bucket].append(row)

plot_resources = []
for bucket in sorted(plot_resource_bins):
    rows = plot_resource_bins[bucket]
    plot_resources.append(
        {
            "minute": rounded(
                mean([row["unix_time"] - resource_start for row in rows]) / 60,
                3,
            ),
            "gpu_mem_mib": rounded(
                max(
                    max(row["gpu0_used_mib"], row["gpu1_used_mib"])
                    for row in rows
                ),
                1,
            ),
            "gpu_util_pct": rounded(
                mean(
                    [
                        (row["gpu0_util_pct"] + row["gpu1_util_pct"]) / 2
                        for row in rows
                    ]
                ),
                1,
            ),
            "rss_gib": rounded(
                max(row["matched_rss_kib"] for row in rows) / 1024**2,
                3,
            ),
            "anon_gib": rounded(
                max(row["cgroup_anon_bytes"] for row in rows) / 1024**3,
                3,
            ),
        }
    )

checkpoint_files = []
for path in sorted(checkpoint_path.rglob("*")):
    if path.is_file():
        checkpoint_files.append(
            {
                "relative_path": str(path.relative_to(checkpoint_path)),
                "bytes": path.stat().st_size,
                "gib": rounded(path.stat().st_size / 1024**3, 3),
            }
        )

started_at = datetime.fromisoformat((runtime / "started_at.txt").read_text().strip())
finished_at = datetime.fromisoformat((runtime / "finished_at.txt").read_text().strip())
wall_seconds = (finished_at - started_at).total_seconds()

result = {
    "snapshot": {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_seconds": wall_seconds,
        "wall_minutes": rounded(wall_seconds / 60, 3),
        "exit_code": int((runtime / "exit_code.txt").read_text().strip()),
        "steps": len(steps),
        "global_batch": 32,
        "sample_presentations": len(steps) * 32,
        "dataset_frames": 7188,
        "epoch_equivalents": rounded(len(steps) * 32 / 7188, 3),
    },
    "training_summary": {
        "loss_step1": losses[0],
        "loss_step2000": losses[-1],
        "loss_first100_mean": rounded(mean(losses[:100])),
        "loss_last100_mean": rounded(mean(losses[-100:])),
        "loss_first100_median": rounded(median(losses[:100])),
        "loss_last100_median": rounded(median(losses[-100:])),
        "loss_min": min(losses),
        "loss_min_step": steps[losses.index(min(losses))],
        "loss_max": max(losses),
        "loss_max_step": steps[losses.index(max(losses))],
        "loss_last100_to_first100_ratio": rounded(
            mean(losses[-100:]) / mean(losses[:100])
        ),
        "rlt_loss_max_abs_diff": max(
            abs(loss - rlt) for loss, rlt in zip(losses, rlt_losses)
        ),
        "vla_loss_abs_max": max(abs(value) for value in vla_losses),
        "grad_mean": rounded(mean(grads)),
        "grad_p50": rounded(percentile(grads, 0.50)),
        "grad_p95": rounded(percentile(grads, 0.95)),
        "grad_max": max(grads),
        "grad_max_step": steps[grads.index(max(grads))],
        "grad_over_1_count": sum(value > 1 for value in grads),
        "grad_over_10_count": sum(value > 10 for value in grads),
        "lr_min": min(lrs),
        "lr_max": max(lrs),
        "lr_max_first_step": steps[lrs.index(max(lrs))],
        "training_time_p50_s": rounded(percentile(train_times[1:], 0.50)),
        "training_time_p95_s": rounded(percentile(train_times[1:], 0.95)),
        "training_time_max_s_excluding_step1": max(train_times[1:]),
        "step2000_total_s": step_times[-1],
        "step2000_training_s": train_times[-1],
        "endpoint_save_overhead_s_approx": rounded(
            step_times[-1] - train_times[-1]
        ),
    },
    "key_metrics": key_metrics,
    "loss_windows": loss_windows,
    "resource_summary": {
        "samples": len(resources),
        "duration_seconds": resources[-1]["unix_time"] - resources[0]["unix_time"],
        "active_samples": len(active_resources),
        "gpu0_peak_mib": max(row["gpu0_used_mib"] for row in resources),
        "gpu1_peak_mib": max(row["gpu1_used_mib"] for row in resources),
        "active_gpu0_util_mean_pct": rounded(
            mean(row["gpu0_util_pct"] for row in active_resources), 3
        ),
        "active_gpu1_util_mean_pct": rounded(
            mean(row["gpu1_util_pct"] for row in active_resources), 3
        ),
        "active_combined_gpu_util_mean_pct": rounded(
            mean(
                (row["gpu0_util_pct"] + row["gpu1_util_pct"]) / 2
                for row in active_resources
            ),
            3,
        ),
        "active_combined_gpu_util_p50_pct": rounded(
            percentile(
                [
                    (row["gpu0_util_pct"] + row["gpu1_util_pct"]) / 2
                    for row in active_resources
                ],
                0.50,
            ),
            3,
        ),
        "matched_rss_peak_gib": rounded(
            max(row["matched_rss_kib"] for row in resources) / 1024**2,
            3,
        ),
        "cgroup_anon_peak_gib": rounded(
            max(row["cgroup_anon_bytes"] for row in resources) / 1024**3,
            3,
        ),
        "cgroup_file_peak_gib": rounded(
            max(row["cgroup_file_bytes"] for row in resources) / 1024**3,
            3,
        ),
        "cgroup_current_peak_gib": rounded(
            max(row["cgroup_current_bytes"] for row in resources) / 1024**3,
            3,
        ),
        "host_available_min_gib": rounded(
            min(row["host_available_bytes"] for row in resources) / 1024**3,
            3,
        ),
        "compute_process_count_max": int(
            max(row["compute_process_count"] for row in resources)
        ),
        "final_gpu0_mib": resources[-1]["gpu0_used_mib"],
        "final_gpu1_mib": resources[-1]["gpu1_used_mib"],
        "final_compute_process_count": int(resources[-1]["compute_process_count"]),
    },
    "checkpoint": {
        "path": str(checkpoint_path),
        "files": checkpoint_files,
        "total_bytes": sum(entry["bytes"] for entry in checkpoint_files),
        "total_gib": rounded(
            sum(entry["bytes"] for entry in checkpoint_files) / 1024**3,
            3,
        ),
        "temporary_residue": [
            str(path.relative_to(checkpoint_path))
            for path in checkpoint_path.rglob("*")
            if "tmp" in path.name.lower()
        ],
    },
    "plot_training": plot_training,
    "plot_resources": plot_resources,
}

print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
PY
