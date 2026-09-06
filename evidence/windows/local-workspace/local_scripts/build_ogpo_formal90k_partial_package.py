#!/usr/bin/env python3
"""Build the compact evidence package for the interrupted OGPO formal-v2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ogpo_formal_live import parse_tensorboard_dump  # noqa: E402
from build_ogpo_formal35k_package import (  # noqa: E402
    base_fragment,
    chart_svg,
    downsample,
    moving_average,
    parse_metric_tables,
    parse_resources,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


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


def parse_audit_checkpoints(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or raw[:200].count(b"\x00") > 20:
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    start_marker = "=== checkpoints ==="
    end_marker = "=== scalar summary ==="
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    payload = text[start:end].strip()
    value, _ = json.JSONDecoder().raw_decode(payload)
    return value


def tensorboard_series(tb: dict, tag: str) -> list[dict]:
    return tb["scalars"].get(tag, [])


def eval_points(tb: dict) -> list[dict]:
    return [
        {
            "rows": int(item["step"]),
            "successes": int(round(float(item["value"]) * 20)),
            "episodes": 20,
            "success_rate": float(item["value"]),
        }
        for item in tensorboard_series(tb, "eval/success_once")
    ]


def build_training_fragment(tb: dict, output: Path) -> None:
    train_raw = tensorboard_series(tb, "env/success_once")
    train = [(float(x["step"]), float(x["value"]) * 100) for x in train_raw]
    avg = moving_average(train, 5)
    evals = eval_points(tb)
    eval_marks = [
        (float(x["rows"]), float(x["success_rate"]) * 100, f'{x["successes"]}/20')
        for x in evals
    ]

    def series(tag: str) -> list[tuple[float, float]]:
        return [
            (float(x["step"]), float(x["value"]))
            for x in tensorboard_series(tb, tag)
        ]

    ratio = series("train/ogpo/ratio")
    bc = series("train/ogpo/bc_loss")
    critic = [(x, min(y, 0.12)) for x, y in series("train/ogpo/critic_loss")]
    q = series("train/ogpo/q_mean")
    td = series("train/ogpo/td_target_mean")
    x_max = 65000.0
    charts = [
        chart_svg(
            label="Policy success through the interrupted 64k run",
            description="Eight-episode EMA rollout waves, five-wave mean, and 20-episode online fixed evaluations.",
            x_max=x_max,
            y_min=0,
            y_max=80,
            y_ticks=[0, 20, 40, 60, 80],
            suffix="%",
            series=[
                ("EMA train wave", "series-1", train, "2 5"),
                ("5-wave mean", "series-3", avg, ""),
            ],
            points=eval_marks,
        ),
        chart_svg(
            label="Online / EMA same-chain ratio",
            description="Mean ratio for each update burst; the dashed line is one. This is not a clip fraction.",
            x_max=x_max,
            y_min=0,
            y_max=1.05,
            y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
            series=[("ratio", "series-1", ratio, "")],
            reference=1.0,
        ),
        chart_svg(
            label="Success BC and critic losses",
            description="The first critic cold-start spike is capped at 0.12 so the stable region remains readable.",
            x_max=x_max,
            y_min=0,
            y_max=0.12,
            y_ticks=[0, 0.03, 0.06, 0.09, 0.12],
            series=[
                ("success BC", "series-1", bc, ""),
                ("critic capped", "series-2", critic, ""),
            ],
        ),
        chart_svg(
            label="Q mean and TD-target mean",
            description="Prediction and bootstrap-target means remain aligned after the initial critic transient.",
            x_max=x_max,
            y_min=-0.4,
            y_max=0.16,
            y_ticks=[-0.4, -0.25, -0.1, 0.05, 0.15],
            series=[("Q", "series-1", q, ""), ("TD target", "series-2", td, "2 5")],
        ),
    ]
    output.write_text(
        base_fragment("ogpo-formal-v2-partial-training", "".join(charts)),
        encoding="utf-8",
        newline="\n",
    )


def build_resource_fragment(
    records: list[dict[str, float]], resources: list[dict[str, float]], output: Path
) -> None:
    sampled = downsample(resources, 700)
    extrema = []
    for key in ("cgroup_current_bytes", "gpu0_used_mib", "gpu1_used_mib", "actor_rss_kib"):
        extrema.append(max(resources, key=lambda row: row.get(key, -1)))
    sampled = sorted(
        {row["unix_time"]: row for row in [*sampled, *extrema]}.values(),
        key=lambda row: row["unix_time"],
    )
    start = sampled[0]["unix_time"]
    duration = sampled[-1]["unix_time"] - start
    x_max = 60000.0

    def scaled(key: str, divisor: float = 1.0) -> list[tuple[float, float]]:
        return [
            ((row["unix_time"] - start) / max(duration, 1) * x_max, row[key] / divisor)
            for row in sampled
        ]

    def time_chart(**kwargs: object) -> str:
        rendered = chart_svg(x_max=x_max, **kwargs)
        replacements = {
            ">0k<": ">0h<",
            ">15k<": f">{duration / 14400:.1f}h<",
            ">30k<": f">{duration / 7200:.1f}h<",
            ">45k<": f">{duration / 4800:.1f}h<",
            ">60k<": f">{duration / 3600:.1f}h<",
        }
        for old, new in replacements.items():
            rendered = rendered.replace(old, new)
        return rendered

    elapsed = records[-1]["elapsed_seconds"]
    rollout = sum(row.get("generate_rollouts", 0.0) for row in records)
    training = sum(row.get("actor/run_training", 0.0) for row in records)
    evaluation = sum(row.get("eval", 0.0) for row in records)
    other = max(0.0, elapsed - rollout - training - evaluation)
    parts = [100 * value / elapsed for value in (rollout, training, evaluation, other)]
    allocation = f"""
<section class="plot-section">
  <div class="plot-label"><span>Useful wall time through the last committed wave</span></div>
  <div class="allocation" style="grid-template-columns:{parts[0]:.3f}fr {parts[1]:.3f}fr {parts[2]:.3f}fr {parts[3]:.3f}fr" role="img" aria-label="Rollout {parts[0]:.1f} percent, paired updates {parts[1]:.1f} percent, evaluation {parts[2]:.1f} percent, other {parts[3]:.1f} percent"><span></span><span></span><span></span><span></span></div>
  <div class="allocation-labels"><span><b>{parts[0]:.1f}%</b> rollout</span><span><b>{parts[1]:.1f}%</b> paired updates</span><span><b>{parts[2]:.1f}%</b> eval</span><span><b>{parts[3]:.1f}%</b> other</span></div>
</section>
"""
    charts = [
        allocation,
        time_chart(
            label="Container and process memory",
            description="Cgroup memory crossed Ray's 228 GiB kill threshold; actor and env curves are summed RSS.",
            y_min=0,
            y_max=245,
            y_ticks=[0, 60, 120, 180, 240],
            suffix=" GiB",
            series=[
                ("cgroup", "series-3", scaled("cgroup_current_bytes", 2**30), ""),
                ("actors RSS", "series-1", scaled("actor_rss_kib", 2**20), ""),
                ("env RSS", "series-2", scaled("env_rss_kib", 2**20), "2 5"),
            ],
            reference=228.0,
        ),
        time_chart(
            label="Physical GPU memory",
            description="Both A800s stayed below 57 GiB despite the later CPU-memory failure.",
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
            description="One-second utilization across rollout, paired updates, evaluation, and the final timeout wait.",
            y_min=0,
            y_max=100,
            y_ticks=[0, 25, 50, 75, 100],
            suffix="%",
            series=[
                ("GPU 0", "series-1", scaled("gpu0_util_pct"), ""),
                ("GPU 1", "series-2", scaled("gpu1_util_pct"), ""),
            ],
        ),
    ]
    output.write_text(
        base_fragment("ogpo-formal-v2-partial-resources", "".join(charts)),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--tb-dump", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    records = parse_metric_tables(snapshot / "metrics.log")
    resources, resource_summary = parse_resources(snapshot / "resources_1s.csv")
    tb = parse_tensorboard_dump(args.tb_dump.resolve())
    checkpoints = parse_audit_checkpoints(args.audit.resolve())
    evals = eval_points(tb)

    clean_tb = snapshot / "tensorboard_scalars.json"
    clean_tb.write_text(json.dumps(tb, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
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
    checkpoint_rows = []
    for item in checkpoints:
        checkpoint_rows.append({
            "step": item["step"],
            "global_online_rows": item["global_online_rows"],
            "policy_version": item["policy_version"],
            "complete": item["complete"],
            "total_bytes": item["total_bytes"],
            "total_gib": item["total_bytes"] / 2**30,
            "dcp_bytes": item["components"].get("actor/dcp_checkpoint", 0),
            "ogpo_sidecar_bytes": item["components"].get("actor/ogpo_components", 0),
        })
    write_csv(
        snapshot / "checkpoint_inventory.tsv",
        checkpoint_rows,
        ["step", "global_online_rows", "policy_version", "complete", "total_bytes", "total_gib", "dcp_bytes", "ogpo_sidecar_bytes"],
    )
    # The name is TSV; replace the comma delimiter emitted above with tabs.
    inventory = snapshot / "checkpoint_inventory.tsv"
    inventory.write_text(inventory.read_text(encoding="utf-8").replace(",", "\t"), encoding="utf-8", newline="\n")

    driver_text = ANSI_RE.sub("", (snapshot / "driver.log").read_text(encoding="utf-8", errors="replace"))
    failure_lines = []
    for line in driver_text.splitlines():
        if any(marker in line for marker in (
            "worker(s) were killed due to the node running low on memory",
            "Workers (tasks / actors) killed due to memory pressure",
            "Watchdog caught collective operation timeout",
            "Exception occurred while running EmbodiedOGPOFSDPPolicy",
            "Exiting main process due to a failure",
        )):
            failure_lines.append(line)
    (snapshot / "failure_excerpt.txt").write_text("\n".join(failure_lines) + "\n", encoding="utf-8", newline="\n")

    latest = records[-1]
    train_success = tensorboard_series(tb, "env/success_once")
    successes = sum(float(item["value"]) * 8 for item in train_success)
    rollout = sum(row.get("generate_rollouts", 0.0) for row in records)
    training = sum(row.get("actor/run_training", 0.0) for row in records)
    evaluation = sum(row.get("eval", 0.0) for row in records)
    elapsed = latest["elapsed_seconds"]
    summary = {
        "status": "failed_partial",
        "exit_code": int((snapshot / "exit_code.txt").read_text().strip()),
        "started_at": (snapshot / "started_at.txt").read_text().strip(),
        "finished_at": (snapshot / "finished_at.txt").read_text().strip(),
        "last_committed_rows": int(latest["rows"]),
        "row_budget": 90000,
        "paired_updates": int(latest["ogpo/actor_updates"]),
        "paired_update_budget": 4000,
        "scheduler_expected_updates": math.floor((latest["rows"] - 10000) * 0.05),
        "logged_train_waves": len(train_success),
        "logged_train_episodes": len(train_success) * 8,
        "logged_train_successes": int(round(successes)),
        "logged_train_success_rate": successes / (len(train_success) * 8),
        "fixed_policy_eval": evals,
        "latest_optimization": {
            key.removeprefix("ogpo/"): latest[key]
            for key in (
                "ogpo/actor_loss", "ogpo/actor_grad_norm", "ogpo/bc_loss",
                "ogpo/critic_loss", "ogpo/critic_grad_norm", "ogpo/q_mean",
                "ogpo/td_target_mean", "ogpo/ratio",
            )
        },
        "useful_wall_seconds": elapsed,
        "monitored_wall_seconds": resource_summary["meta"]["duration_seconds"],
        "wall_allocation": {
            "rollout_seconds": rollout,
            "paired_training_seconds": training,
            "evaluation_seconds": evaluation,
            "other_useful_seconds": max(0.0, elapsed - rollout - training - evaluation),
            "post_last_metric_seconds": resource_summary["meta"]["duration_seconds"] - elapsed,
        },
        "resources": {
            "gpu0_peak_mib": resource_summary["gpu0_used_mib"]["max"],
            "gpu1_peak_mib": resource_summary["gpu1_used_mib"]["max"],
            "gpu0_mean_util_pct": resource_summary["gpu0_util_pct"]["mean"],
            "gpu1_mean_util_pct": resource_summary["gpu1_util_pct"]["mean"],
            "cgroup_peak_bytes": resource_summary["cgroup_current_bytes"]["max"],
            "disk_available_min_bytes": resource_summary["disk_available_bytes"]["min"],
            "kernel_oom_events": resource_summary["meta"]["oom_max"],
            "kernel_oom_kill_events": resource_summary["meta"]["oom_kill_max"],
            "ray_kill_threshold_gib": 228.0,
            "cgroup_limit_gib": 240.0,
        },
        "checkpoints": checkpoint_rows,
        "failure_chain": [
            "Ray memory monitor saw 228.21/240.00 GiB and killed a channel worker.",
            "Collective channels broke; the remaining rank waited for its peer.",
            "NCCL timed out after 1800 seconds; driver exited 255.",
        ],
    }
    (snapshot / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    (snapshot / "METRIC_GLOSSARY.md").write_text(
        "# Metric glossary\n\n"
        "- `env/success_once`: eight-episode training rollout wave from the EMA actor.\n"
        "- `eval/success_once`: 20-episode fixed evaluation from the online actor.\n"
        "- `ratio`: burst mean of the online/EMA same-chain likelihood ratio; it is not a clip fraction.\n"
        "- `replay_rows` and `success_rows`: per-rank means; global progress is `total_online_rows`.\n"
        "- cgroup `oom=0`: no kernel OOM; Ray killed a worker proactively at its 95% threshold.\n",
        encoding="utf-8",
        newline="\n",
    )
    (snapshot / "README.md").write_text(
        "# OGPO pi0 RoboTwin formal v2: interrupted 64k evidence package\n\n"
        "This is a lightweight, read-only evidence package. The fresh 90k run stopped after "
        "64,078 committed primitive rows and 2,703 paired updates. Ray killed a worker when "
        "container memory reached 228.21/240 GiB; the later NCCL timeout was a consequence.\n\n"
        "The package contains logs, TensorBoard scalars, resolved configuration, provenance, "
        "one-second resources, two small checkpoint completion manifests, derived tables, and "
        "plots. It contains no checkpoint tensors or replay payloads. The 30,959-row and "
        "60,968-row full checkpoints remain on the server and are complete.\n",
        encoding="utf-8",
        newline="\n",
    )

    args.visual_dir.mkdir(parents=True, exist_ok=True)
    visuals = snapshot / "visuals"
    visuals.mkdir(exist_ok=True)
    training_html = args.visual_dir / "ogpo-formal-v2-partial-training.html"
    resource_html = args.visual_dir / "ogpo-formal-v2-partial-resources.html"
    build_training_fragment(tb, training_html)
    build_resource_fragment(records, resources, resource_html)
    shutil.copy2(training_html, visuals / "training-fragment.html")
    shutil.copy2(resource_html, visuals / "resources-fragment.html")
    for png_name in (
        "ogpo-formal-v2-partial-training.png",
        "ogpo-formal-v2-partial-resources.png",
    ):
        png = args.visual_dir / png_name
        if png.is_file():
            shutil.copy2(png, visuals / png.name)

    manifest = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        if path.name == "PACKAGE_MANIFEST.json":
            continue
        manifest.append({
            "path": path.relative_to(snapshot).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    (snapshot / "PACKAGE_MANIFEST.json").write_text(
        json.dumps({"files": manifest}, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
            archive.write(path, path.relative_to(snapshot.parent))
    args.zip.with_suffix(args.zip.suffix + ".sha256").write_text(
        f"{sha256(args.zip)}  {args.zip.name}\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
