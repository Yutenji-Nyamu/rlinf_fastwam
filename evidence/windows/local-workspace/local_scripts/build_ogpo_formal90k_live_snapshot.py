#!/usr/bin/env python3
"""Build a compact live evidence snapshot for the active OGPO 90k run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ogpo_formal35k_package import (  # noqa: E402
    base_fragment,
    chart_svg,
    downsample,
    moving_average,
    parse_metric_tables,
    parse_resources,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluation_points(records: list[dict[str, float]]) -> list[dict[str, float]]:
    points = [{"rows": 0.0, "successes": 1.0, "episodes": 20.0, "success_rate": 0.05}]
    for record in records:
        if "eval_success_once" not in record:
            continue
        rate = record["eval_success_once"]
        points.append(
            {
                "rows": record["rows"],
                "successes": round(rate * 20),
                "episodes": 20.0,
                "success_rate": rate,
            }
        )
    return points


def build_training_fragment(
    records: list[dict[str, float]], evals: list[dict[str, float]], output: Path
) -> None:
    current_rows = records[-1]["rows"]
    x_max = max(22000.0, math.ceil(current_rows / 1000) * 1000)
    train = [(r["rows"], r["train_success_once"] * 100) for r in records]
    avg = moving_average(train, 5)
    eval_marks = [
        (p["rows"], p["success_rate"] * 100, f'{int(p["successes"])}/20') for p in evals
    ]
    learned = [r for r in records if "ogpo/ratio" in r]
    ratio = [(r["rows"], r["ogpo/ratio"]) for r in learned]
    bc = [(r["rows"], r["ogpo/bc_loss"]) for r in learned]
    critic = [(r["rows"], min(r["ogpo/critic_loss"], 0.12)) for r in learned]
    q = [(r["rows"], r["ogpo/q_mean"]) for r in learned]
    td = [(r["rows"], r["ogpo/td_target_mean"]) for r in learned]

    charts = [
        chart_svg(
            label="Policy success through the live 21.8k snapshot",
            description="Per-wave train success, five-wave moving average, and fixed-policy evaluations.",
            x_max=x_max,
            y_min=0,
            y_max=55,
            y_ticks=[0, 10, 20, 30, 40, 50],
            suffix="%",
            series=[
                ("train wave", "series-1", train, "2 5"),
                ("5-wave mean", "series-3", avg, ""),
            ],
            points=eval_marks,
        ),
        chart_svg(
            label="Online / EMA same-chain ratio",
            description="Mean probability ratio for each update burst. One is the identity reference.",
            x_max=x_max,
            y_min=0,
            y_max=1.05,
            y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
            series=[("ratio", "series-1", ratio, "")],
            reference=1.0,
        ),
        chart_svg(
            label="Success BC and critic losses",
            description="The first critic cold-start spike is capped at 0.12 so the stable region remains visible.",
            x_max=x_max,
            y_min=0,
            y_max=0.12,
            y_ticks=[0, 0.03, 0.06, 0.09, 0.12],
            series=[
                ("success BC", "series-1", bc, ""),
                ("critic (spike capped)", "series-2", critic, ""),
            ],
        ),
        chart_svg(
            label="Q mean and TD-target mean",
            description="The critic prediction mean follows its bootstrapped target mean after cold start.",
            x_max=x_max,
            y_min=-0.4,
            y_max=0.15,
            y_ticks=[-0.4, -0.25, -0.1, 0.05, 0.15],
            series=[("Q", "series-1", q, ""), ("TD target", "series-2", td, "2 5")],
        ),
    ]
    output.write_text(
        base_fragment("ogpo-formal-v2-live-training", "".join(charts)),
        encoding="utf-8",
        newline="\n",
    )


def build_resource_fragment(
    records: list[dict[str, float]], resources: list[dict[str, float]], output: Path
) -> None:
    sampled = downsample(resources, 650)
    extrema = [
        max(resources, key=lambda r: r.get("cgroup_current_bytes", -1)),
        max(resources, key=lambda r: r.get("gpu0_used_mib", -1)),
        max(resources, key=lambda r: r.get("gpu1_used_mib", -1)),
    ]
    sampled = sorted(
        {r["unix_time"]: r for r in [*sampled, *extrema]}.values(),
        key=lambda r: r["unix_time"],
    )
    start = sampled[0]["unix_time"]
    duration_h = (sampled[-1]["unix_time"] - start) / 3600
    scale_max = 90000.0

    def scaled(key: str, divisor: float = 1.0) -> list[tuple[float, float]]:
        return [
            ((r["unix_time"] - start) / (duration_h * 3600) * scale_max, r[key] / divisor)
            for r in sampled
        ]

    def time_chart(**kwargs: object) -> str:
        rendered = chart_svg(x_max=scale_max, **kwargs)
        replacements = {
            ">0k<": ">0h<",
            ">22.5k<": f">{duration_h / 4:.1f}h<",
            ">45k<": f">{duration_h / 2:.1f}h<",
            ">67.5k<": f">{duration_h * 3 / 4:.1f}h<",
            ">90k<": f">{duration_h:.1f}h<",
        }
        for old, new in replacements.items():
            rendered = rendered.replace(old, new)
        return rendered

    elapsed = records[-1]["elapsed_seconds"]
    rollout = sum(r.get("generate_rollouts", 0.0) for r in records)
    training = sum(r.get("actor/run_training", 0.0) for r in records)
    evaluation = sum(r.get("eval", 0.0) for r in records)
    other = max(0.0, elapsed - rollout - training - evaluation)
    parts = [100 * value / elapsed for value in (rollout, training, evaluation, other)]
    allocation = f"""
<section class="plot-section">
  <div class="plot-label"><span>Measured wall time to the 21.8k snapshot</span></div>
  <div class="allocation" style="grid-template-columns:{parts[0]:.3f}fr {parts[1]:.3f}fr {parts[2]:.3f}fr {parts[3]:.3f}fr" role="img" aria-label="Rollout {parts[0]:.1f} percent, paired updates {parts[1]:.1f} percent, evaluation {parts[2]:.1f} percent, other {parts[3]:.1f} percent"><span></span><span></span><span></span><span></span></div>
  <div class="allocation-labels"><span><b>{parts[0]:.1f}%</b> rollout</span><span><b>{parts[1]:.1f}%</b> paired updates</span><span><b>{parts[2]:.1f}%</b> eval</span><span><b>{parts[3]:.1f}%</b> other</span></div>
</section>
"""
    charts = [
        allocation,
        time_chart(
            label="Physical GPU memory",
            description="One-second physical memory usage on both A800 GPUs.",
            y_min=0,
            y_max=64,
            y_ticks=[0, 16, 32, 48, 64],
            suffix=" GiB",
            series=[
                ("GPU 0", "series-1", scaled("gpu0_used_mib", 1024), ""),
                ("GPU 1", "series-2", scaled("gpu1_used_mib", 1024), ""),
            ],
        ),
        time_chart(
            label="GPU utilization",
            description="One-second utilization of both A800 GPUs across rollout, updates, and evaluation.",
            y_min=0,
            y_max=100,
            y_ticks=[0, 25, 50, 75, 100],
            suffix="%",
            series=[
                ("GPU 0", "series-1", scaled("gpu0_util_pct"), ""),
                ("GPU 1", "series-2", scaled("gpu1_util_pct"), ""),
            ],
        ),
        time_chart(
            label="Container memory",
            description="Cgroup memory grows with the online replay; no checkpoint serialization has occurred yet.",
            y_min=0,
            y_max=160,
            y_ticks=[0, 40, 80, 120, 160],
            suffix=" GiB",
            series=[("cgroup", "series-3", scaled("cgroup_current_bytes", 2**30), "")],
        ),
    ]
    output.write_text(
        base_fragment("ogpo-formal-v2-live-resources", "".join(charts)),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    records = parse_metric_tables(snapshot / "metrics.log")
    resources, resource_summary = parse_resources(snapshot / "resources_1s.csv")
    evals = evaluation_points(records)

    metrics_fields = [
        "wave", "rows", "elapsed_seconds", "train_success_once", "eval_success_once",
        "ogpo/actor_updates", "ogpo/critic_updates", "ogpo/updates_run", "ogpo/ratio",
        "ogpo/actor_loss", "ogpo/actor_grad_norm", "ogpo/bc_loss", "ogpo/critic_loss",
        "ogpo/critic_grad_norm", "ogpo/q_mean", "ogpo/td_target_mean",
        "generate_rollouts", "actor/run_training", "eval", "step_seconds",
    ]
    write_csv(snapshot / "metrics_table.csv", records, metrics_fields)
    write_csv(snapshot / "evaluation_points.csv", evals, ["rows", "successes", "episodes", "success_rate"])
    (snapshot / "resource_summary.json").write_text(
        json.dumps(resource_summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )

    latest = records[-1]
    elapsed = latest["elapsed_seconds"]
    rollout = sum(r.get("generate_rollouts", 0.0) for r in records)
    training = sum(r.get("actor/run_training", 0.0) for r in records)
    evaluation = sum(r.get("eval", 0.0) for r in records)
    successes = sum(r["train_success_once"] * 8 for r in records)
    summary = {
        "snapshot_at": "2026-08-08T18:08:08+08:00",
        "status": "running",
        "rows": latest["rows"],
        "row_budget": 90000,
        "paired_updates": latest.get("ogpo/actor_updates", 0),
        "paired_update_budget": 4000,
        "waves": len(records),
        "train_episodes": len(records) * 8,
        "train_successes": successes,
        "train_success_rate": successes / (len(records) * 8),
        "fixed_policy_eval": evals,
        "latest_optimization": {
            key.removeprefix("ogpo/"): latest[key]
            for key in (
                "ogpo/actor_loss", "ogpo/actor_grad_norm", "ogpo/bc_loss",
                "ogpo/critic_loss", "ogpo/critic_grad_norm", "ogpo/q_mean",
                "ogpo/td_target_mean", "ogpo/ratio",
            )
        },
        "wall_seconds": elapsed,
        "wall_allocation": {
            "rollout_seconds": rollout,
            "paired_training_seconds": training,
            "evaluation_seconds": evaluation,
            "other_seconds": max(0.0, elapsed - rollout - training - evaluation),
        },
        "resources": {
            "gpu0_peak_mib": resource_summary["gpu0_used_mib"]["max"],
            "gpu1_peak_mib": resource_summary["gpu1_used_mib"]["max"],
            "gpu0_mean_util_pct": resource_summary["gpu0_util_pct"]["mean"],
            "gpu1_mean_util_pct": resource_summary["gpu1_util_pct"]["mean"],
            "cgroup_peak_bytes": resource_summary["cgroup_current_bytes"]["max"],
            "host_available_min_bytes": resource_summary["host_available_bytes"]["min"],
            "disk_available_min_bytes": resource_summary["disk_available_bytes"]["min"],
            "oom_events": resource_summary["meta"]["oom_max"],
            "oom_kill_events": resource_summary["meta"]["oom_kill_max"],
        },
        "checkpoint": {"count": 0, "next_due_rows": 30000},
    }
    (snapshot / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )

    readme = """# OGPO pi0 RoboTwin formal v2: live 21.8k snapshot

Read-only snapshot captured while the 90k run was still active. It contains only light logs,
TensorBoard/config provenance, the full one-second resource trace to the snapshot, and derived
tables/plots. It contains no replay or checkpoint tensor.

- Progress: 21,802/90,000 replay-valid primitive rows; 590/4,000 paired updates.
- Fixed-policy eval: 1/20 at row 0, 1/20 near 10k, 3/20 near 20k.
- No checkpoint exists yet; the first full resume checkpoint is due at 30k.
- GPU peaks: 57,231/56,834 MiB; cgroup peak 155,821,133,824 bytes; OOM/OOM-kill 0.
"""
    (snapshot / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    args.visual_dir.mkdir(parents=True, exist_ok=True)
    visuals = snapshot / "visuals"
    visuals.mkdir(exist_ok=True)
    training_html = args.visual_dir / "ogpo-formal-v2-live-training.html"
    resource_html = args.visual_dir / "ogpo-formal-v2-live-resources.html"
    build_training_fragment(records, evals, training_html)
    build_resource_fragment(records, resources, resource_html)
    shutil.copy2(training_html, visuals / "training-fragment.html")
    shutil.copy2(resource_html, visuals / "resources-fragment.html")

    manifest = []
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        if path.name == "PACKAGE_MANIFEST.json":
            continue
        manifest.append(
            {"path": path.relative_to(snapshot).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    (snapshot / "PACKAGE_MANIFEST.json").write_text(
        json.dumps({"files": manifest}, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(snapshot.parent))
    args.zip.with_suffix(args.zip.suffix + ".sha256").write_text(
        f"{sha256(args.zip)}  {args.zip.name}\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
